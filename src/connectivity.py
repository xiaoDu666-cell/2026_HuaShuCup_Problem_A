from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict
from .geometry import (
    Cylinder, cylinder_surface_distance, segment_plane_distance_to_x_plane,
    Sphere, sphere_sphere_surface_distance, sphere_cylinder_surface_distance,
    SphericalCap, 
    sphericalcap_sphericalcap_surface_distance, 
    sphericalcap_cylinder_surface_distance,
    sphericalcap_sphere_surface_distance
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


def particle_surface_distance(p1, p2, mode='fast'):
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
        sphericalcap_sphere_surface_distance
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
        if mode == 'fast':
            return sphere_sphere_surface_distance(
                Sphere(p1.c, p1.r), Sphere(p2.c, p2.r)
            )
        else:
            return sphericalcap_sphericalcap_surface_distance(p1, p2)
    
    # SphericalCap - Sphere
    if isinstance(p1, SphericalCap) and isinstance(p2, Sphere):
        if mode == 'fast':
            return sphere_sphere_surface_distance(Sphere(p1.c, p1.r), p2)
        else:
            return sphericalcap_sphere_surface_distance(p1, p2)
    if isinstance(p1, Sphere) and isinstance(p2, SphericalCap):
        if mode == 'fast':
            return sphere_sphere_surface_distance(p1, Sphere(p2.c, p2.r))
        else:
            return sphericalcap_sphere_surface_distance(p2, p1)
    
    # SphericalCap - Cylinder
    if isinstance(p1, SphericalCap) and isinstance(p2, Cylinder):
        if mode == 'fast':
            return sphere_cylinder_surface_distance(Sphere(p1.c, p1.r), p2)
        else:
            return sphericalcap_cylinder_surface_distance(p1, p2)
    if isinstance(p1, Cylinder) and isinstance(p2, SphericalCap):
        if mode == 'fast':
            return sphere_cylinder_surface_distance(Sphere(p2.c, p2.r), p1)
        else:
            return sphericalcap_cylinder_surface_distance(p2, p1)
    
    # 默认回退
    c1 = p1.center() if hasattr(p1, 'center') else np.zeros(3)
    c2 = p2.center() if hasattr(p2, 'center') else np.zeros(3)
    return max(0.0, np.linalg.norm(c1 - c2) - 1000)


def build_connectivity(particles: List, L: float, thresh: float = 1.8, 
                       left_plane_x: float = None, right_plane_x: float = None,
                       mode: str = 'fast') -> Tuple[bool, UnionFind]:
    """
    构建粒子连通性。
    
    mode:
        'fast': 粗扫模式，球冠用完整球近似（速度快）
        'exact': 精扫模式，使用精确计算（速度慢但准确）
    """
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
        if hasattr(s, 'aabb'):
            aabbs.append(s.aabb())
        else:
            aabbs.append((np.array([-np.inf, -np.inf, -np.inf]), 
                          np.array([np.inf, np.inf, np.inf])))

    # ============================================================
    # 距离计算循环（已修复：取消注释，添加 AABB 容差）
        # ============================================================
    for i in range(N):
        lo_i, hi_i = aabbs[i]
        for j in range(i + 1, N):
            lo_j, hi_j = aabbs[j]
            if np.any(lo_i > hi_j + 1e-12) or np.any(lo_j > hi_i + 1e-12):
                continue
            dist = particle_surface_distance(particles[i], particles[j], mode=mode)
            #print(f"DEBUG: i={i}, j={j}, dist={dist:.6f}")  # 强制输出
            if dist <= thresh + 1e-9:
                uf.union(i, j)
                #print(f"DEBUG: 合并 {i} 和 {j}")

    # ============================================================
    # 电极接触判定
    # ============================================================
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
                #print(f"DEBUG: 球 {s.id} 接触左电极, c={s.c}, r={s.r}, surf_left={surf_left}")
                uf.union(i, LEFT_NODE)
            d_right = abs(s.c[0] - right_plane_x)
            surf_right = max(0.0, d_right - s.r)
            if surf_right <= thresh + 1e-9:
                #print(f"DEBUG: 球 {s.id} 接触右电极, c={s.c}, r={s.r}, surf_right={surf_right}")
                uf.union(i, RIGHT_NODE)
                
        elif isinstance(s, SphericalCap):
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


def are_indices_connected(uf, idx_a: int, idx_b: int) -> bool:
    return uf.find(idx_a) == uf.find(idx_b)


def are_ids_connected(uf, particles: List, id_a: int, id_b: int) -> bool:
    idx_a = next((i for i, s in enumerate(particles) if getattr(s, 'id', None) == id_a), None)
    idx_b = next((i for i, s in enumerate(particles) if getattr(s, 'id', None) == id_b), None)
    if idx_a is None or idx_b is None:
        return False
    return are_indices_connected(uf, idx_a, idx_b)