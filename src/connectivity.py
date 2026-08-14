from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict
from .geometry import Cylinder, cylinder_surface_distance, segment_plane_distance_to_x_plane
from .geometry import Sphere, sphere_sphere_surface_distance, sphere_cylinder_surface_distance

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
    shifted = (point + L/2.0) / cell_size
    return tuple(np.floor(shifted).astype(int))

def build_connectivity(particles: List, L: float, thresh: float = 1.8, left_plane_x: float = None, right_plane_x: float = None) -> Tuple[bool, UnionFind]:
    """
    particles: list of objects, each either Cylinder segments (Cylinder) or Sphere instances.
    For cylinders that were split, pass the resulting segments (each with .id).
    """
    if left_plane_x is None:
        left_plane_x = -L/2.0
    if right_plane_x is None:
        right_plane_x = L/2.0

    uf = UnionFind()
    LEFT_NODE = ("PLANE", "LEFT")
    RIGHT_NODE = ("PLANE", "RIGHT")
    uf.find(LEFT_NODE); uf.find(RIGHT_NODE)

    # Build id->indices mapping for same original id unioning
    id_to_indices: Dict[int, List[int]] = {}
    for idx, p in enumerate(particles):
        pid = getattr(p, 'id', None)
        id_to_indices.setdefault(pid, []).append(idx)
    for inds in id_to_indices.values():
        first = inds[0]
        for other in inds[1:]:
            uf.union(first, other)

    # spatial hashing
    radii = []
    centers = []
    aabbs = []
    for s in particles:
        if isinstance(s, Cylinder):
            r = s.r
            c = s.center()
            lo, hi = s.aabb()
        elif isinstance(s, Sphere):
            r = s.r
            c = s.center()
            lo, hi = s.aabb()
        else:
            # unknown type; skip
            continue
        radii.append(r)
        centers.append(c)
        aabbs.append((lo, hi))

    max_r = max(radii) if radii else 0.0
    cell_size = max((2 * max_r + thresh), L / 20.0)
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for i, c in enumerate(centers):
        idx = _grid_index(c, cell_size, L)
        grid.setdefault(idx, []).append(i)

    neighs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    N = len(particles)
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
                if np.any(lo_i > hi_j) or np.any(lo_j > hi_i):
                    continue
                p_i = particles[i]
                p_j = particles[j]
                # dispatch distance based on types
                if isinstance(p_i, Cylinder) and isinstance(p_j, Cylinder):
                    dist = cylinder_surface_distance(p_i, p_j)
                elif isinstance(p_i, Sphere) and isinstance(p_j, Sphere):
                    dist = sphere_sphere_surface_distance(p_i, p_j)
                else:
                    # mixed
                    if isinstance(p_i, Sphere) and isinstance(p_j, Cylinder):
                        dist = sphere_cylinder_surface_distance(p_i, p_j)
                    else:
                        dist = sphere_cylinder_surface_distance(p_j, p_i)
                if dist <= thresh + 1e-9:
                    uf.union(i, j)

    # connect to electrodes (planes)
    for i, s in enumerate(particles):
        if isinstance(s, Cylinder):
            d_left = segment_plane_distance_to_x_plane(s.p0, s.p1, left_plane_x)
            surf_left = max(0.0, d_left - s.r)
            if surf_left <= thresh + 1e-9:
                uf.union(i, LEFT_NODE)
            d_right = segment_plane_distance_to_x_plane(s.p0, s.p1, right_plane_x)
            surf_right = max(0.0, d_right - s.r)
            if surf_right <= thresh + 1e-9:
                uf.union(i, RIGHT_NODE)
        elif isinstance(s, Sphere):
            d_left = abs(s.c[0] - left_plane_x)
            surf_left = max(0.0, d_left - s.r)
            if surf_left <= thresh + 1e-9:
                uf.union(i, LEFT_NODE)
            d_right = abs(s.c[0] - right_plane_x)
            surf_right = max(0.0, d_right - s.r)
            if surf_right <= thresh + 1e-9:
                uf.union(i, RIGHT_NODE)

    connected = (uf.find(LEFT_NODE) == uf.find(RIGHT_NODE))
    return connected, uf

# helper queries
def are_indices_connected(uf, idx_a: int, idx_b: int) -> bool:
    return uf.find(idx_a) == uf.find(idx_b)

def are_ids_connected(uf, particles: List, id_a: int, id_b: int) -> bool:
    idx_a = next((i for i, s in enumerate(particles) if getattr(s, 'id', None) == id_a), None)
    idx_b = next((i for i, s in enumerate(particles) if getattr(s, 'id', None) == id_b), None)
    if idx_a is None or idx_b is None:
        return False
    return are_indices_connected(uf, idx_a, idx_b)