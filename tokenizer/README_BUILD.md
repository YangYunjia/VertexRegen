# Building the patched `vertexregen_tokenizer` extension

Local patch adding `pinned_vertices` to `edge_collapse_with_record`, so the QEM
collapse can be driven down to the 8-vertex fastener-panel seed instead of the
topological floor. Needed by the seeded PM arms (`vr_seed_*`).

Upstream `build.sh` is cibuildwheel/manylinux and irrelevant here. `cmake` is not
installed in `armesh`, so scikit-build is not an option either — but CGAL 6 is
header-only, so a direct link works.

## What the patch does

| file | change |
|---|---|
| `src/pinned_vertex.h` | new — `Pin_map`, `Pinned_vertex_filter`, `Pinned_vertex_placement` |
| `src/simplify.cpp` | `pinned_vertices` arg, `resolve_pins()`, unconditional placement composition |
| `src/visitor.cpp` | migrate the pin to the survivor in `OnCollapsed` |
| `src/{simplify.h,pybind.cpp,common.h}` | thread the arg through; expose `Stats.num_pinned` |

Three things are load-bearing and easy to get wrong:

1. **Pin vertices, not edges.** CGAL rejects a collapse when *both* endpoints
   touch a constrained edge (`Edge_collapse.h:823`), so `edge_is_constrained_map`
   over a vertex's incident edges freezes its whole one-ring — the mesh floors at
   the union of the pinned stars (21–25 verts on our brackets), keeping fine fan
   geometry. Vetoing only pin-to-pin collapses lets a pinned vertex keep absorbing
   its unpinned neighbours.
2. **The filter cannot set the placement.** `Edge_collapse.h:633` only tests the
   filter's return against `nullopt` and then collapses to the *original*
   placement, so vetoing and repositioning need two separate objects.
3. **Pins must migrate.** `Euler::collapse_edge` always erases `source` and never
   flips the halfedge to spare a vertex, so a pinned endpoint can be the one
   removed. Without the `OnCollapsed` migration only 4–6 of 8 corners survive to
   the base mesh; with it, 8/8.

The patch also fixes a pre-existing bug at `simplify.cpp:99`, where
`auto placement = cond ? constrained : bounded` sliced `Constrained_placement`
down to its base — that layer was a no-op even with `sharp_angle_threshold > 0`.

## Rebuild

```bash
conda activate armesh
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
cd experiments/baselines/vertexregen/tokenizer

$CXX -O3 -shared -std=c++17 -fPIC -fvisibility=hidden \
  $(python -m pybind11 --includes) \
  -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/eigen3 \
  src/pybind.cpp src/simplify.cpp src/mesh.cpp src/visitor.cpp \
  -lgmp -lgmpxx -lmpfr -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib \
  -o _vertexregen_tokenizer_pybind.cpython-312-x86_64-linux-gnu.so
```

~37 s. Install into the dep dir the eval driver already prepends:

```bash
cp _vertexregen_tokenizer_pybind.cpython-312-x86_64-linux-gnu.so \
   /home/yunjia/.deps_vertexregen/vertexregen_tokenizer/
```

The pre-patch build is kept beside it as `*.so.orig` — restore it to reproduce
the original published-baseline column.

## Usage

```python
st = VR.edge_collapse_with_record(
    Vq, F,
    8, 6,                       # target V, target F -- the stop predicate ANDs
    True,                       # no_placement: survivor snaps to an endpoint
    -1, False,                  # sharp_angle_threshold, strict
    record_full_info=True,
    pinned_vertices=[0, 1, 2, 3, 4, 5, 6, 7],
)
assert st.num_pinned == 8       # all pins resolved in the cleaned mesh
```

`pinned_vertices` indexes the **input** array. `polygon_soup_to_mesh` may merge
and reorder vertices, so `resolve_pins` maps them back by exact point equality —
sound because the coordinates are quantized integers held in doubles.

The stop predicate is an `AND` of both counts
(`simplify_stop_predicate.h:38`); for a triangulated polygon use `F = V - 2`.
Passing two small "plausible" numbers collapses straight past the target.

Omitting `pinned_vertices` reproduces upstream behaviour exactly.

## Verification

Run after any rebuild:

- **Unpinned parity.** With no pins, collapse count, sequence length and a hash
  over the full `(v_s, v_t, placement)` sequence must match the `.so.orig` on the
  same brackets. Load each build in its own process — importing the extension
  both top-level and through the package raises
  `generic_type: type "PolygonSoup" is already registered!`.
- **Pinned termination.** On bracket v2 train, `num_pos_tokens=256`,
  `pinned_vertices=[0..7]`: measured **148/150 (98.67%)** reach V=8/F=6 with all
  8 panel corners present; 2 stall above V=8; **0** lose a corner. For contrast,
  the same target without pinning keeps only 0–3 of the 8 corners.
