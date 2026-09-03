"""Mesh npz (``create_mesh_pkl.py --kind mesh``) -> VertexRegen vsplit HF dataset.

Expects per-sample ``verts``, ``faces``, ``seed_verts``, ``seed_faces`` (and
optionally ``condition``, which is *not* written -- Ship-D params / raw panels
are not position tokens). The quantised seed is stored as ``seed_mesh_q`` for
downstream comparison with ``init_vertices`` and, when present, as the optional
coordinate-token prefix in ``vertexregen.data.VertexRegenCollator``.

Collapse is pinned to the seed by default (``pinned_vertices`` resolved by
quantised coordinate match). Pass ``--no_pin`` only to reproduce upstream's
free QEM floor (targets 3/1).

Replaces the three prior paths:
  * ``preprocess_vertexregen.py``           -- raw HF + panels_q / condition_q
  * ``preprocess_vertexregen_seeded.py``     -- pinned collapse, bracket-only
  * ``preprocess_mesh_npz.py``            -- hull raw HF, no seed / collapse

Usage (hull v5):
    PYTHONPATH=experiments/baselines/vertexregen:. \\
      python experiments/baselines/vertexregen/scripts/preprocess_vsplit_dataset.py \\
        --train_npz data/shipd/hullv5_mesh/train.npz \\
        --val_npz   data/shipd/hullv5_mesh/val.npz \\
        --out experiments/baselines/vertexregen/data/hullv5_vsplit_q256 \\
        --num_pos_tokens 256 --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED_MESH_KEY = "seed_mesh_q"



def _pin_indices(Vq: np.ndarray, seed_q: np.ndarray) -> Optional[List[int]]:
    """Map each seed quantised point to an index in the full mesh grid."""
    first = {}
    for i, v in enumerate(Vq):
        t = tuple(map(int, v))
        if t not in first:
            first[t] = i
    pins = []
    for s in seed_q:
        t = tuple(map(int, s))
        if t not in first:
            return None
        pins.append(first[t])
    return pins


def encode_one(payload):
    """Worker: one mesh -> one vsplit row, or a drop reason."""
    (idx, V, F, Sv, Sf, num_pos_tokens, frame, pin, screen_selfint, uid_prefix) = payload
    try:
        from vertexregen_tokenizer.collapse import (
            CollapseError,
            quantized_edge_collapse,
            validate_vertex_split_sequence,
        )
        from vertexregen_tokenizer.utils import quantize_points  # noqa: F401
        from vertexregen.decode import cleanup_mesh

        V = np.asarray(V, np.float64)
        F = np.asarray(F, np.int32)
        Sv = np.asarray(Sv, np.float64).reshape(-1, 3)
        Sf = np.asarray(Sf, np.int32).reshape(-1, 3)

        if screen_selfint:
            # Optional; bracket meshes can self-intersect. Soft-fail if helper missing.
            try:
                root = Path(__file__).resolve().parents[3]
                sys.path.insert(0, str(root / "experiments/bracket"))
                from mesh_quality import mesh_quality

                si = float(
                    mesh_quality(V, F.astype(np.int64)).get(
                        "self_intersection_ratio", 0.0
                    )
                )
                if si > 0.0:
                    return idx, None, "self_intersecting_gt"
            except ImportError:
                pass

        # ONE corpus-constant, per-axis grid for the mesh, the seed AND the
        # collapse. Upstream quantises each mesh by its own bbox, which makes the
        # grid a function of the target: unusable at generation time, where the
        # target does not exist. The grid must also be shared with the collapse
        # -- `init_vertices` comes back on whatever grid it used and is compared
        # against `seed_q` below, so two different grids reject every sample
        # ("base_not_seed", measured 34883/35215 when they disagreed).
        #
        # Measured against upstream on 800 meshes: per-axis uses 242/221/246 of
        # 256 levels (upstream fills only each mesh's longest axis) and collides
        # LESS often, 0.165% vs 0.266%.
        Vq = frame.encode(V)
        seed_q = frame.encode(Sv).astype(int)
        n_seed_v, n_seed_f = int(seed_q.shape[0]), int(Sf.shape[0])

        if pin:
            pins = _pin_indices(Vq, seed_q)
            if pins is None:
                return idx, None, "seed_not_in_mesh"
            target_v, target_f = n_seed_v, n_seed_f
        else:
            pins = []
            # Upstream free-floor targets; the AND-ed stop predicate rarely hits
            # them exactly, so collapse runs to the topological floor.
            target_v, target_f = 3, 1

        try:
            res = quantized_edge_collapse(
                V,
                F,
                num_pos_tokens,
                quantized_vertices=Vq,   # same grid as Vq/seed_q above
                pinned_vertices=pins,
                target_number_of_vertices=target_v,
                target_number_of_triangles=target_f,
            )
        except CollapseError as err:
            return idx, None, err.reason

        init_v = np.asarray(res.init_vertices, int)
        init_f = np.asarray(res.init_faces, int)
        if pin:
            if len(init_v) != n_seed_v or len(init_f) != n_seed_f:
                return idx, None, f"base_{len(init_v)}v_{len(init_f)}f"
            want = {tuple(map(int, v)) for v in seed_q}
            got = {tuple(map(int, v)) for v in init_v}
            if want != got:
                return idx, None, "base_not_seed"

        if len(res.vsplit_seq) == 0:
            return idx, None, "empty_vsplit_seq"
        if not validate_vertex_split_sequence(res):
            return idx, None, "roundtrip_failed"
        _, cleaned_init_faces = cleanup_mesh(init_v, init_f)
        if len(cleaned_init_faces) != len(init_f):
            return idx, None, "degenerate_base"

        verts = np.asarray(res.vertices, int)
        vmap = {tuple(v): i for i, v in enumerate(verts)}
        row = dict(
            uid=f"{uid_prefix}_{idx:06d}",
            vertices=verts,
            faces=np.asarray(res.faces, int),
            init_vertices_ref=[vmap[tuple(v)] for v in init_v],
            init_faces=init_f,
            vsplit_seq=np.asarray(res.vsplit_seq, int),
            # Quantised seed in the mesh frame -- for comparing init_vertices
            # and for the optional coordinate-token prefix at train time.
            seed_mesh_q=seed_q,
        )
        return idx, row, "ok"
    except Exception as err:  # noqa: BLE001
        return idx, None, f"{type(err).__name__}: {str(err)[:60]}"


def run_split(npz_path: str, split: str, args, frame) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    required = ("verts", "faces", "seed_verts", "seed_faces")
    missing = [k for k in required if k not in d.files]
    if missing:
        raise SystemExit(
            f"{npz_path} missing {missing}; need create_mesh_pkl.py --kind mesh "
            f"output (keys: verts/faces/seed_verts/seed_faces[/condition])."
        )

    t_load = time.time()
    verts, faces = d["verts"], d["faces"]
    seed_v, seed_f = d["seed_verts"], d["seed_faces"]

    # encode() clips silently, so an out-of-frame sample would be corrupted
    # geometry rather than an error. The frame is fitted on train with a margin;
    # verify it on every split instead of trusting the margin.
    bad = sum(int(frame.out_of_range(np.asarray(v, np.float64))) for v in verts)
    if bad:
        raise SystemExit(
            f"[{split}] {bad} sample(s) fall outside the quantisation frame. "
            f"Refit on the union of splits or raise the margin -- never clip.")
    n_total = len(verts)
    n = n_total if args.limit is None else min(args.limit, n_total)
    print(f"  [{split}] loaded {n_total} samples in {time.time()-t_load:.1f}s", flush=True)

    jobs = []
    for i in range(n):
        F = np.asarray(faces[i], np.int32)
        if args.max_faces is not None and len(F) > args.max_faces:
            continue
        jobs.append(
            (
                i,
                np.asarray(verts[i], np.float64),
                F,
                np.asarray(seed_v[i], np.float64),
                np.asarray(seed_f[i], np.int32),
                args.num_pos_tokens,
                frame,
                not args.no_pin,
                not args.keep_selfint,
                split,
            )
        )
    skipped_faces = n - len(jobs)
    if skipped_faces:
        print(f"  [{split}] skipped {skipped_faces} over max_faces={args.max_faces}",
              flush=True)

    rows: List[Tuple[int, Dict[str, Any]]] = []
    reasons: Dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(encode_one, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            idx, row, why = fut.result()
            reasons[why] = reasons.get(why, 0) + 1
            if row is not None:
                rows.append((idx, row))
            if k % 250 == 0 or k == len(jobs):
                el = time.time() - t0
                ok = reasons.get("ok", 0)
                print(
                    f"  [{split}] {k}/{len(jobs)}  ok={ok} ({100*ok/max(k,1):.2f}%)  "
                    f"{k/max(el,1e-9):.1f}/s  "
                    f"eta {(len(jobs)-k)/max(k/el,1e-9)/60:.1f}m",
                    flush=True,
                )
    rows.sort(key=lambda r: r[0])
    print(f"  [{split}] done: {len(rows)}/{len(jobs)} kept; reasons={reasons}", flush=True)
    return dict(rows=[r for _, r in rows], n_input=len(jobs), reasons=reasons)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train_npz", type=Path, required=True)
    ap.add_argument("--val_npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_pos_tokens", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_faces", type=int, default=None)
    ap.add_argument(
        "--no_pin",
        action="store_true",
        help="free QEM base (upstream 3/1 targets); default pins to seed_mesh",
    )
    ap.add_argument(
        "--keep_selfint",
        action="store_true",
        help="do not drop self-intersecting GT (needs experiments/bracket/mesh_quality)",
    )
    args = ap.parse_args()

    import datasets

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"num_pos_tokens={args.num_pos_tokens} pin={not args.no_pin} "
        f"screen_selfint={not args.keep_selfint} workers={args.workers}",
        flush=True,
    )

    # ONE frame for the whole corpus, fitted on TRAIN only and reused for val.
    # A per-sample frame keyed on the target mesh would leak the answer's centre
    # and extent to this arm and not to the projection arms; see
    # optshapegen/preprocess/quant_frame.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from optshapegen.preprocess.quant_frame import QuantFrame, fit_frame

    _tr = np.load(str(args.train_npz), allow_pickle=True)["verts"]
    frame = fit_frame((np.asarray(v, np.float64) for v in _tr), args.num_pos_tokens)
    print(f"quant frame (per-axis): centre={[round(c,5) for c in frame.centre]} "
          f"half={[round(h,5) for h in frame.half]} npos={frame.num_pos_tokens}",
          flush=True)

    t0 = time.time()
    res = {
        "train": run_split(str(args.train_npz), "train", args, frame),
        "test": run_split(str(args.val_npz), "test", args, frame),
    }
    dd = datasets.DatasetDict(
        {k: datasets.Dataset.from_list(v["rows"]) for k, v in res.items()}
    )
    dd.save_to_disk(str(out))

    meta = {
        "format": "vertexregen_vsplit_seeded",
        "source_train": str(args.train_npz),
        "source_val": str(args.val_npz),
        "num_pos_tokens": args.num_pos_tokens,
        **frame.to_meta(),
        "pin_to_seed": not args.no_pin,
        "seed_mesh_key": SEED_MESH_KEY,
        "screened_self_intersecting_gt": not args.keep_selfint,
        "elapsed_sec": time.time() - t0,
        "splits": {
            k: {
                "n_input": v["n_input"],
                "n_kept": len(v["rows"]),
                "reasons": v["reasons"],
            }
            for k, v in res.items()
        },
    }
    with open(out / "encode_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["splits"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
