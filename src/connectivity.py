from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict
# from .geometry import Cylinder, cylinder_surface_distance, segment_plane_distance_to_x_plane
# from .geometry import Sphere, sphere_sphere_surface_distance, sphere_cylinder_surface_distance
# from .geometry import SphericalCap, sphericalcap_sphericalcap_surface_distance, sphericalcap_cylinder_surface_distance
from .geometry import (
    Cylinder, cylinder_surface_distance, segment_plane_distance_to_x_plane,
    Sphere, sphere_sphere_surface_distance, sphere_cylinder_surface_distance,
    SphericalCap, 
    sphericalcap_sphericalcap_surface_distance, 
    sphericalcap_cylinder_surface_distance,
    sphericalcap_sphere_surface_distance  # 新增
)


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
    if left_plane_x is None:
        left_plane_x = -L / 2.0
    if right_plane_x is None:
        right_plane_x = L / 2.0

    uf = UnionFind()
    LEFT_NODE = ("PLANE", "LEFT")
    RIGHT_NODE = ("PLANE", "RIGHT")
    uf.find(LEFT_NODE)
    uf.find(RIGHT_NODE)

    N = len(particles)

    # 预计算 AABB
    aabbs = []
    for s in particles:
        if isinstance(s, Cylinder) or isinstance(s, Sphere):
            aabbs.append(s.aabb())
        else:
            aabbs.append((np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])))

    # 暴力 O(N²) 距离计算
    for i in range(N):
        lo_i, hi_i = aabbs[i]
        for j in range(i + 1, N):
            lo_j, hi_j = aabbs[j]
            # AABB 快速排除
            if np.any(lo_i > hi_j) or np.any(lo_j > hi_i):
                continue
            # 精确距离计算
            # 精确距离计算（支持 Cylinder, Sphere, SphericalCap）
            dist = particle_surface_distance(particles[i], particles[j])
            if dist <= thresh + 1e-9:
                uf.union(i, j)
            # p_i = particles[i]
            # p_j = particles[j]
            # if isinstance(p_i, Cylinder) and isinstance(p_j, Cylinder):
            #     dist = cylinder_surface_distance(p_i, p_j)
            # elif isinstance(p_i, Sphere) and isinstance(p_j, Sphere):
            #     dist = sphere_sphere_surface_distance(p_i, p_j)
            # else:
            #     if isinstance(p_i, Sphere) and isinstance(p_j, Cylinder):
            #         dist = sphere_cylinder_surface_distance(p_i, p_j)
            #     else:
            #         dist = sphere_cylinder_surface_distance(p_j, p_i)
            # if dist <= thresh + 1e-9:
            #     uf.union(i, j)

    # 电极接触判定
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

        elif isinstance(s, SphericalCap):
        # 球冠与电极的距离：近似为球心到电极平面的距离 - 球冠高度方向上的投影
        # 简单近似：使用球心到电极平面的距离 - 半径（保守估计）
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

# 在 connectivity.py 中添加

def particle_surface_distance(p1, p2, L=None):
    """
    计算两个粒子之间的表面最短距离。
    支持类型：Cylinder, Sphere, SphericalCap
    """
    from .geometry import (
        Cylinder, Sphere, SphericalCap,
        cylinder_surface_distance,
        sphere_sphere_surface_distance,
        sphere_cylinder_surface_distance,
        sphericalcap_sphericalcap_surface_distance,
        sphericalcap_cylinder_surface_distance,
        sphericalcap_sphere_surface_distance  # 新增
    )
    
    # Cylinder - Cylinder
    if isinstance(p1, Cylinder) and isinstance(p2, Cylinder):
        return cylinder_surface_distance(p1, p2)
    
    # Sphere - Sphere
    if isinstance(p1, Sphere) and isinstance(p2, Sphere):
        return sphere_sphere_surface_distance(p1, p2)
    
    # Sphere - Cylinder
    if isinstance(p1, Sphere) and isinstance(p2, Cylinder):
        return sphere_cylinder_surface_distance(p1, p2)
    if isinstance(p1, Cylinder) and isinstance(p2, Sphere):
        return sphere_cylinder_surface_distance(p2, p1)
    
    # SphericalCap - SphericalCap
    if isinstance(p1, SphericalCap) and isinstance(p2, SphericalCap):
        return sphericalcap_sphericalcap_surface_distance(p1, p2)
    
    # SphericalCap - Sphere（精确计算）
    if isinstance(p1, SphericalCap) and isinstance(p2, Sphere):
        return sphericalcap_sphere_surface_distance(p1, p2)
    if isinstance(p1, Sphere) and isinstance(p2, SphericalCap):
        return sphericalcap_sphere_surface_distance(p2, p1)
    
    # SphericalCap - Cylinder
    if isinstance(p1, SphericalCap) and isinstance(p2, Cylinder):
        return sphericalcap_cylinder_surface_distance(p1, p2)
    if isinstance(p1, Cylinder) and isinstance(p2, SphericalCap):
        return sphericalcap_cylinder_surface_distance(p2, p1)
    
    # 默认：使用中心距离（保守估计）
    c1 = p1.center() if hasattr(p1, 'center') else np.zeros(3)
    c2 = p2.center() if hasattr(p2, 'center') else np.zeros(3)
    return max(0.0, np.linalg.norm(c1 - c2) - 1000)