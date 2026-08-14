from __future__ import annotations
import numpy as np
from typing import Tuple

def _clamp(v, a, b):
    return max(a, min(b, v))

class Cylinder:
    """Represent a finite circular cylinder by its axis endpoints and radius.

    Attributes:
        p0, p1: numpy arrays shape (3,) for axis endpoints
        r: radius (float)
        id: optional identifier linking back to original particle
    """

    def __init__(self, p0: np.ndarray, p1: np.ndarray, r: float, id: int = None):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.r = float(r)
        self.id = id

    def axis_vector(self) -> np.ndarray:
        return self.p1 - self.p0

    def length(self) -> float:
        return np.linalg.norm(self.axis_vector())

    def center(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        # Axis-aligned bounding box including radius
        lo = np.minimum(self.p0, self.p1) - self.r
        hi = np.maximum(self.p0, self.p1) + self.r
        return lo, hi

class Sphere:
    """Represent a sphere by center and radius.

    Attributes:
        c: center (3,)
        r: radius
        id: optional identifier
    """
    def __init__(self, center: np.ndarray, r: float, id: int = None):
        self.c = np.asarray(center, dtype=float)
        self.r = float(r)
        self.id = id

    def center(self) -> np.ndarray:
        return self.c

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.c - self.r
        hi = self.c + self.r
        return lo, hi

def _seg_seg_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Compute minimal distance between two line segments [p1,q1] and [p2,q2]."""
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    D = a * c - b * b
    SMALL_NUM = 1e-12

    sc, sN, sD = 0.0, 0.0, D
    tc, tN, tD = 0.0, 0.0, D

    if D < SMALL_NUM:  # almost parallel
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD

    dP = w0 + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

def cylinder_surface_distance(c1: Cylinder, c2: Cylinder) -> float:
    """Return minimal surface-to-surface distance between two cylinders (>=0)."""
    da = _seg_seg_distance(c1.p0, c1.p1, c2.p0, c2.p1)
    surf = da - (c1.r + c2.r)
    return max(0.0, surf)

def sphere_sphere_surface_distance(s1: Sphere, s2: Sphere) -> float:
    d = np.linalg.norm(s1.c - s2.c)
    return max(0.0, d - (s1.r + s2.r))

def sphere_cylinder_surface_distance(s: Sphere, c: Cylinder) -> float:
    # distance from sphere center to closest point on cylinder axis segment minus radii
    p1 = c.p0
    p2 = c.p1
    v = p2 - p1
    w = s.c - p1
    vv = np.dot(v, v)
    if vv == 0:
        proj = p1
    else:
        t = np.dot(w, v) / vv
        t_clamped = max(0.0, min(1.0, t))
        proj = p1 + t_clamped * v
    dcenter = np.linalg.norm(s.c - proj)
    return max(0.0, dcenter - (s.r + c.r))

def segment_plane_distance_to_x_plane(p0: np.ndarray, p1: np.ndarray, x_plane: float) -> float:
    x0 = p0[0]
    x1 = p1[0]
    if (x0 - x_plane) * (x1 - x_plane) <= 0:
        return 0.0
    else:
        return min(abs(x0 - x_plane), abs(x1 - x_plane))