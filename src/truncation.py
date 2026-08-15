from __future__ import annotations
import numpy as np
from typing import List
from .geometry import Cylinder, Sphere, SphericalCap

def _map_point_into_box(pt: np.ndarray, L: float) -> np.ndarray:
    """Map a point into the primary box [-L/2, L/2]^3 using periodic wrapping."""
    half = L / 2.0
    mapped = ((pt + half) % L) - half
    return mapped

def split_cylinder_by_box(cyl: Cylinder, L: float) -> List[Cylinder]:
    """
    Existing implementation: split axis at faces, shift segments so their midpoint lies in box.
    """
    # (original implementation kept — for brevity assume it's present here)
    # We'll reuse the original function body from your repo.
    A = np.asarray(cyl.p0, dtype=float)
    B = np.asarray(cyl.p1, dtype=float)
    d = B - A
    half = L / 2.0
    SMALL = 1e-15
    ts = [0.0, 1.0]
    for dim in range(3):
        a_coord = A[dim]
        b_coord = B[dim]
        min_coord = min(a_coord, b_coord)
        max_coord = max(a_coord, b_coord)
        if abs(d[dim]) < SMALL:
            continue
        n_min = int(np.floor((min_coord + half) / L)) - 1
        n_max = int(np.ceil((max_coord + half) / L)) + 1
        face_positions = []
        for n in range(n_min, n_max + 1):
            face_positions.append(-half + n * L)
            face_positions.append(half + n * L)
        for face in face_positions:
            denom = (B[dim] - A[dim])
            if abs(denom) < SMALL:
                continue
            t = (face - A[dim]) / denom
            if 0.0 < t < 1.0:
                ts.append(float(t))
    ts = sorted(set(ts))
    segments = []
    for i in range(len(ts) - 1):
        t0 = ts[i]
        t1 = ts[i + 1]
        if t1 - t0 <= 1e-15:
            continue
        p0 = A + d * t0
        p1 = A + d * t1
        midpoint = 0.5 * (p0 + p1)
        mapped_mid = _map_point_into_box(midpoint, L)
        shift = mapped_mid - midpoint
        p0_shift = p0 + shift
        p1_shift = p1 + shift
        p0_shift = np.minimum(np.maximum(p0_shift, -half), half)
        p1_shift = np.minimum(np.maximum(p1_shift, -half), half)
        segments.append(Cylinder(p0_shift, p1_shift, cyl.r, id=cyl.id))
    return segments

# 在 truncation.py 中重写 split_sphere_by_box

def split_sphere_by_box(sph: Sphere, L: float) -> List:
    """
    将球按照边界截断规则切割，返回盒内的所有部分。
    与圆柱的处理逻辑一致：超出边界的部分平移到另一侧。
    """
    from .geometry import SphericalCap
    
    half = L / 2.0
    c = sph.c
    r = sph.r
    
    # 检查球是否完全在盒内
    if np.all(c - r >= -half) and np.all(c + r <= half):
        return [sph]
    
    # 球心映射到盒内
    mapped_c = _map_point_into_box(c, L)
    
    # 检查映射后的球是否完全在盒内
    if np.all(mapped_c - r >= -half) and np.all(mapped_c + r <= half):
        return [Sphere(mapped_c, r, id=sph.id)]
    
    parts = []
    
    # 检查每个方向
    for dim in range(3):
        # ---- 负方向越界：mapped_c[dim] - r < -half ----
        if mapped_c[dim] - r < -half:
            # 盒内部分：平面 x_dim = -half，保留 x_dim >= -half 一侧
            # 法向量指向正方向，d = (-half - center) · 1 = -half - center
            n = np.zeros(3)
            n[dim] = 1.0
            d = -half - mapped_c[dim]
            parts.append(SphericalCap(mapped_c, r, n, d, id=sph.id))
            
            # 平移回来的部分：从另一侧（+half 附近）进入
            shifted_c = mapped_c.copy()
            shifted_c[dim] = mapped_c[dim] + L
            # 法向量指向负方向，d = (half - shifted_center) · (-1) = shifted_center - half
            n2 = np.zeros(3)
            n2[dim] = -1.0
            d2 = shifted_c[dim] - half
            parts.append(SphericalCap(shifted_c, r, n2, d2, id=sph.id))
        
        # ---- 正方向越界：mapped_c[dim] + r > half ----
        if mapped_c[dim] + r > half:
            # 盒内部分：平面 x_dim = half，保留 x_dim <= half 一侧
            # 法向量指向负方向，d = (half - center) · (-1) = center - half
            n = np.zeros(3)
            n[dim] = -1.0
            d = mapped_c[dim] - half
            parts.append(SphericalCap(mapped_c, r, n, d, id=sph.id))
            
            # 平移回来的部分：从另一侧（-half 附近）进入
            shifted_c = mapped_c.copy()
            shifted_c[dim] = mapped_c[dim] - L
            # 法向量指向正方向，d = (-half - shifted_center) · 1 = -half - shifted_center
            n2 = np.zeros(3)
            n2[dim] = 1.0
            d2 = -half - shifted_c[dim]
            parts.append(SphericalCap(shifted_c, r, n2, d2, id=sph.id))
    
    # 如果没有产生任何部分，返回映射后的完整球
    if not parts:
        return [Sphere(mapped_c, r, id=sph.id)]
    
    # 去重并过滤空球冠
    valid_parts = []
    seen = set()
    for p in parts:
        if p.is_empty():
            continue
        key = (tuple(np.round(p.c, 6)), tuple(np.round(p.n, 6)), round(p.d, 6))
        if key not in seen:
            seen.add(key)
            valid_parts.append(p)
    
    return valid_parts if valid_parts else [Sphere(mapped_c, r, id=sph.id)]