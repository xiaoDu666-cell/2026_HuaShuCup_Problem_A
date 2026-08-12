# connectivity.py
# Union-Find and connectivity builder using spatial hashing to reduce pair checks.

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict
from .geometry import Cylinder, cylinder_surface_distance, segment_plane_distance_to_x_plane

class UnionFind:
    def __init__(self):
        self.parent = dict()

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        self.parent[rb] = ra

def _grid_index(point: np.ndarray, cell_size: float, L: float):
    """Return integer grid index tuple for a point in box [-L/2, L/2].
    Uses floor((point + L/2)/cell_size) to handle negative coords consistently.
    """
    shifted = (point + L/2.0) / cell_size
    return tuple(np.floor(shifted).astype(int))

def build_connectivity(segments: List[Cylinder],
                       L: float,
                       thresh: float = 1.8,
                       left_plane_x: float = None,
                       right_plane_x: float = None) -> Tuple[bool, UnionFind]:
    """Build connectivity graph among segments. Returns (is_connected, uf).
    segments: list of Cylinder segments (each with .id identifying original particle)
    left_plane_x, right_plane_x: x coordinate of left and right electrodes; default to -L/2, +L/2
    """
    if left_plane_x is None:
        left_plane_x = -L/2.0
    if right_plane_x is None:
        right_plane_x = L/2.0

    uf = UnionFind()
    LEFT_NODE = ("PLANE", "LEFT")
    RIGHT_NODE = ("PLANE", "RIGHT")
    uf.find(LEFT_NODE); uf.find(RIGHT_NODE)

    # ensure all segments that belong to same original particle are unioned
    id_to_indices: Dict[int, List[int]] = {}
    for idx, seg in enumerate(segments):
        id_to_indices.setdefault(seg.id, []).append(idx)

    for inds in id_to_indices.values():
        first = inds[0]
        for other in inds[1:]:
            uf.union(first, other)

    # spatial hashing based on segment centers
    radii = [s.r for s in segments] if segments else [0.0]
    max_r = max(radii)
    cell_size = max((2 * max_r + thresh), L / 20.0)  # heuristic
    grid: Dict[Tuple[int, int, int], List[int]] = {}

    aabbs = []
    centers = []
    for i, s in enumerate(segments):
        lo, hi = s.aabb()
        aabbs.append((lo, hi))
        centers.append(s.center())
        idx = _grid_index(centers[-1], cell_size, L)
        grid.setdefault(idx, []).append(i)

    # neighbor offsets to check (27 neighborhood)
    neighs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    N = len(segments)
    for i in range(N):
        ci = centers[i]
        gi = _grid_index(ci, cell_size, L)
        lo_i, hi_i = aabbs[i]
        for d in neighs:
            gj = (gi[0] + d[0], gi[1] + d[1], gi[2] + d[2])
            for j in grid.get(gj, []):
                if j <= i:
                    continue
                lo_j, hi_j = aabbs[j]
                # AABB no-overlap cull: if any axis lo_i > hi_j or lo_j > hi_i then disjoint
                if np.any(lo_i > hi_j) or np.any(lo_j > hi_i):
                    continue
                dist = cylinder_surface_distance(segments[i], segments[j])
                if dist <= thresh + 1e-9:
                    uf.union(i, j)

    # connect to planes
    for i, s in enumerate(segments):
        d_left = segment_plane_distance_to_x_plane(s.p0, s.p1, left_plane_x)
        surf_left = max(0.0, d_left - s.r)
        if surf_left <= thresh + 1e-9:
            uf.union(i, LEFT_NODE)
        d_right = segment_plane_distance_to_x_plane(s.p0, s.p1, right_plane_x)
        surf_right = max(0.0, d_right - s.r)
        if surf_right <= thresh + 1e-9:
            uf.union(i, RIGHT_NODE)

    connected = (uf.find(LEFT_NODE) == uf.find(RIGHT_NODE))
    return connected, uf

# --- 辅助查询函数 (可追加到 src/connectivity.py 文件末尾) ---

def are_indices_connected(uf, idx_a: int, idx_b: int) -> bool:
    """判断两个 segment 索引是否属于同一连通集合（使用并查集 uf）。"""
    return uf.find(idx_a) == uf.find(idx_b)

def are_ids_connected(uf, segments: List[Cylinder], id_a: int, id_b: int) -> bool:
    """判断原始粒子 id_a 与 id_b 是否连通（segments 列表中查找对应的 segment 索引）。"""
    idx_a = next((i for i, s in enumerate(segments) if s.id == id_a), None)
    idx_b = next((i for i, s in enumerate(segments) if s.id == id_b), None)
    if idx_a is None or idx_b is None:
        return False
    return are_indices_connected(uf, idx_a, idx_b)