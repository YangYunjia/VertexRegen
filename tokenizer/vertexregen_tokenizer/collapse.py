import numpy as np
from collections import namedtuple
from typing import Optional, Sequence

from ._vertexregen_tokenizer_pybind import edge_collapse_with_record, vertex_split
from .utils import normalize_vertices, quantize_points, sort_quantized_mesh


CollapseResult = namedtuple(
    "CollapseResult",
    [
        "init_vertices",
        "init_faces",
        "vsplit_seq",
        "vertices",
        "faces",
        "vsplit_result_seq",
    ],
)


class CollapseError(Exception):
    """Raised when quantised edge collapse or record derivation fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def quantized_edge_collapse(
    vertices,
    faces,
    num_pos_tokens,
    *,
    pinned_vertices: Optional[Sequence[int]] = None,
    target_number_of_vertices: int = 3,
    target_number_of_triangles: int = 1,
    quantized_vertices: Optional[np.ndarray] = None,
) -> CollapseResult:
    """Run quantised QEM edge collapse and derive the vertex-split record.

    With default stop targets (3 vertices / 1 triangle) and no pins, behaviour
    matches upstream VertexRegen. Pass ``pinned_vertices`` and explicit stop
    counts to collapse to a seeded base mesh instead of the topological floor.

    Raises ``CollapseError`` on failure; inspect ``exc.reason`` for the cause
    (e.g. ``"invalid_input"``, ``"pins_unresolved"``).

    ``quantized_vertices`` overrides the internal quantisation. Upstream always
    normalises each mesh by its OWN bounding box. Pass a constant quantizition
    grid instead to make the frame available at inference and identical across
    samples.

    Everything after this point operates on integer coordinates, so supplying
    them directly changes nothing else about the procedure.
    """
    if quantized_vertices is None:
        quantized_vertices = quantize_points(
            normalize_vertices(np.array(vertices)), num_pos_tokens
        )
    else:
        quantized_vertices = np.asarray(quantized_vertices)
        if quantized_vertices.shape != (len(np.asarray(vertices)), 3):
            raise CollapseError("invalid_input")
    faces = np.array(faces)
    pinned = list(pinned_vertices or [])

    stats = edge_collapse_with_record(
        quantized_vertices,
        faces,
        target_number_of_vertices=target_number_of_vertices,
        target_number_of_triangles=target_number_of_triangles,
        no_placement=True,  # survivor snaps to an endpoint: no new coords
        sharp_angle_threshold=-1,
        strict=True,
        record_full_info=True,
        pinned_vertices=pinned,
    )

    if not stats.is_valid:
        raise CollapseError("invalid_input")
    if pinned and int(stats.num_pinned) != len(pinned):
        raise CollapseError("pins_unresolved")

    vertices = np.array(stats.cleaned_mesh.vertices).astype(int)
    faces = np.array(stats.cleaned_mesh.faces).astype(int)
    vertex_map = {tuple(v): i for i, v in enumerate(vertices)}
    collapse_info = []
    result_seq = [(vertices.copy(), faces.copy())]
    for item in stats.collapse_sequence:
        v_s = tuple(item.v_s_p.astype(int))
        v_t = tuple(item.v_t_p.astype(int))
        v_placement = tuple(item.v_placement.astype(int))
        v_s = vertex_map.get(v_s, -1)
        v_t = vertex_map.get(v_t, -1)
        v_placement = vertex_map.get(v_placement, -1)
        if v_s == -1 or v_t == -1 or v_placement == -1:
            raise CollapseError("vertex_mapping_failed")
        v_l = item.v_l_p
        if v_l is not None:
            v_l = tuple(v_l.astype(int))
            v_l = vertex_map.get(v_l, -1)
        else:
            v_l = -1
        v_r = item.v_r_p
        if v_r is not None:
            v_r = tuple(v_r.astype(int))
            v_r = vertex_map.get(v_r, -1)
        else:
            v_r = -1
        if v_l == -1 and v_r == -1:
            raise CollapseError("both_wings_nil")
        if v_placement == v_t:
            v_l, v_r = v_r, v_l
            v_s, v_t = v_t, v_s
        collapse_info.append([v_s, v_l, v_r, v_t])
        result_seq.append(
            (
                np.array(item.collapsed_mesh.vertices).astype(int),
                np.array(item.collapsed_mesh.faces).astype(int),
            )
        )

    collapse_info = np.array(collapse_info[::-1]).astype(int)
    result_seq = result_seq[::-1]
    init_vertices, init_faces = result_seq[0]
    return CollapseResult(
        init_vertices=init_vertices,
        init_faces=init_faces,
        vsplit_seq=collapse_info,
        vertices=vertices,
        faces=faces,
        vsplit_result_seq=result_seq[1:],
    )


def edge_collapse_quantized_mesh(v, f, v_s, v_t):
    # Collapse v_t into v_s
    f[f == v_t] = v_s
    # Swap v_t with the last vertex to remove
    if v_t != len(v) - 1:
        f[f == len(v) - 1] = v_t
        v[v_t] = v[-1]
    v = v[:-1]
    # Remove degenerate faces
    face_mask = np.logical_or.reduce(
        (f[:, 0] == f[:, 1], f[:, 1] == f[:, 2], f[:, 2] == f[:, 0])
    )
    f = f[~face_mask]
    return v, f


def compare_quantized_mesh(v1, f1, v2, f2):
    if v1.shape != v2.shape or f1.shape != f2.shape:
        return False
    v1, f1 = sort_quantized_mesh(v1, f1)
    v2, f2 = sort_quantized_mesh(v2, f2)
    if not np.equal(v1, v2).all():
        return False
    if not np.equal(f1, f2).all():
        return False
    return True


def validate_edge_collapse_sequence(info: CollapseResult):
    all_vertices = np.array(info.vertices).astype(int)
    current_vertices = all_vertices.copy()
    current_faces = np.array(info.faces).astype(int)
    collapse_seq = info.vsplit_seq[::-1]
    gt_results = [
        (info.init_vertices.astype(int), info.init_faces.astype(int))
    ] + info.vsplit_result_seq
    gt_results = gt_results[::-1][1:]
    for s_idx, (v_s, v_l, v_r, v_t) in enumerate(collapse_seq):
        if v_l == -1 and v_r == -1:
            return False
        v_s_p = tuple(all_vertices[v_s])
        v_t_p = tuple(all_vertices[v_t])
        v_mapping = {tuple(v): i for i, v in enumerate(current_vertices)}
        v_s_i = v_mapping.get(v_s_p, -1)
        v_t_i = v_mapping.get(v_t_p, -1)
        if v_s_i == -1 or v_t_i == -1:
            return False
        num_vertices = current_vertices.shape[0]
        current_vertices, current_faces = edge_collapse_quantized_mesh(
            current_vertices, current_faces, v_s_i, v_t_i
        )
        new_num_vertices = current_vertices.shape[0]
        if new_num_vertices != num_vertices - 1:
            raise RuntimeError("Vertex count mismatch after collapse")
        mesh_equals = compare_quantized_mesh(
            current_vertices,
            current_faces,
            gt_results[s_idx][0],
            gt_results[s_idx][1],
        )
        if not mesh_equals:
            return False
    return True


def validate_vertex_split_sequence(info: CollapseResult):
    curr_vertices = np.array(info.init_vertices).astype(int)
    curr_faces = np.array(info.init_faces).astype(int)
    all_vertices = np.array(info.vertices).astype(int)
    vsplit_seq = info.vsplit_seq
    current_vertex_mapping = {tuple(v): i for i, v in enumerate(curr_vertices)}
    for s_idx, (v_s, v_l, v_r, v_t) in enumerate(vsplit_seq):
        v_s_p = tuple(all_vertices[v_s])
        v_l_p = tuple(all_vertices[v_l]) if v_l != -1 else None
        v_r_p = tuple(all_vertices[v_r]) if v_r != -1 else None
        v_t_p = tuple(all_vertices[v_t])
        v_s_i = current_vertex_mapping.get(v_s_p, -1)
        v_l_i = current_vertex_mapping.get(v_l_p, -1) if v_l_p is not None else None
        v_r_i = current_vertex_mapping.get(v_r_p, -1) if v_r_p is not None else None
        if v_s_i == -1 or v_l_i == -1 or v_r_i == -1:
            return False
        if v_t_p in current_vertex_mapping:
            return False
        gt_vertices, gt_faces = info.vsplit_result_seq[s_idx]
        try:
            result = vertex_split(curr_vertices, curr_faces, v_s_i, v_l_i, v_r_i, v_t_p)
            if result is None:
                return False
        except RuntimeError:
            return False
        curr_vertices = np.array(result.vertices).astype(int)
        curr_faces = np.array(result.faces).astype(int)
        mesh_equals = compare_quantized_mesh(
            curr_vertices, curr_faces, gt_vertices, gt_faces
        )
        if not mesh_equals:
            return False
        current_vertex_mapping = {tuple(v): i for i, v in enumerate(curr_vertices)}
    return True
