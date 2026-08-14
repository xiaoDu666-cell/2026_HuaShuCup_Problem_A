from __future__ import annotations
import numpy as np
from typing import List
from .geometry import Cylinder, Sphere

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

def split_sphere_by_box(sph: Sphere, L: float) -> List[Sphere]:
    """
    For spheres, periodic mapping of center into primary box suffices (we model whole sphere
    translated so its center lies inside the box). Return a single Sphere instance shifted.
    """
    mapped_center = _map_point_into_box(sph.c, L)
    # clamp numerics
    half = L / 2.0
    mapped_center = np.minimum(np.maximum(mapped_center, -half), half)
    return [Sphere(mapped_center, sph.r, id=sph.id)]