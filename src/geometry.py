from __future__ import annotations
import numpy as np
from typing import Tuple

# Try to import numba; if unavailable, provide a no-op decorator so code runs without numba.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    def njit(*args, **kwargs):
        def _decorator(f):
            return f
        return _decorator
    NUMBA_AVAILABLE = False


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

# ---------- Python implementations (fallback) ----------

def _seg_seg_distance_py(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
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
    return float(np.linalg.norm(dP))


def cylinder_surface_distance_py(c1: Cylinder, c2: Cylinder) -> float:
    da = _seg_seg_distance_py(c1.p0, c1.p1, c2.p0, c2.p1)
    surf = da - (c1.r + c2.r)
    return float(max(0.0, surf))


def sphere_sphere_surface_distance_py(s1: Sphere, s2: Sphere) -> float:
    d = float(np.linalg.norm(s1.c - s2.c))
    return float(max(0.0, d - (s1.r + s2.r)))


def sphere_cylinder_surface_distance_py(s: Sphere, c: Cylinder) -> float:
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
    dcenter = float(np.linalg.norm(s.c - proj))
    return float(max(0.0, dcenter - (s.r + c.r)))

# ---------- numba-accelerated implementations (if available) ----------
if NUMBA_AVAILABLE:
    @njit
    def _seg_seg_distance_numba(p1, q1, p2, q2):
        u0 = q1[0] - p1[0]; u1 = q1[1] - p1[1]; u2 = q1[2] - p1[2]
        v0 = q2[0] - p2[0]; v1 = q2[1] - p2[1]; v2 = q2[2] - p2[2]
        w0_0 = p1[0] - p2[0]; w0_1 = p1[1] - p2[1]; w0_2 = p1[2] - p2[2]
        a = u0*u0 + u1*u1 + u2*u2
        b = u0*v0 + u1*v1 + u2*v2
        c = v0*v0 + v1*v1 + v2*v2
        d = u0*w0_0 + u1*w0_1 + u2*w0_2
        e = v0*w0_0 + v1*w0_1 + v2*w0_2
        D = a*c - b*b
        SMALL_NUM = 1e-12
        sN = 0.0; sD = D
        tN = 0.0; tD = D
        if D < SMALL_NUM:
            sN = 0.0
            sD = 1.0
            tN = e
            tD = c
        else:
            sN = (b*e - c*d)
            tN = (a*e - b*d)
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
                sN = 0.0
            elif (-d + b) > a:
                sN = sD
            else:
                sN = (-d + b)
                sD = a
        sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
        tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD
        dP0 = w0_0 + sc*u0 - tc*v0
        dP1 = w0_1 + sc*u1 - tc*v1
        dP2 = w0_2 + sc*u2 - tc*v2
        return (dP0*dP0 + dP1*dP1 + dP2*dP2)**0.5

    @njit
    def cylinder_surface_distance_numba(p0, p1, r1, q0, q1, r2):
        da = _seg_seg_distance_numba(p0, p1, q0, q1)
        surf = da - (r1 + r2)
        if surf < 0.0:
            return 0.0
        return surf

    @njit
    def sphere_sphere_surface_distance_numba(c1, r1, c2, r2):
        dx = c1[0]-c2[0]; dy = c1[1]-c2[1]; dz = c1[2]-c2[2]
        d = (dx*dx + dy*dy + dz*dz)**0.5
        surf = d - (r1 + r2)
        if surf < 0.0:
            return 0.0
        return surf

    @njit
    def sphere_cylinder_surface_distance_numba(sc, sr, p0, p1, cr):
        # project sc onto segment p0-p1
        vx = p1[0]-p0[0]; vy = p1[1]-p0[1]; vz = p1[2]-p0[2]
        wx = sc[0]-p0[0]; wy = sc[1]-p0[1]; wz = sc[2]-p0[2]
        vv = vx*vx + vy*vy + vz*vz
        if vv == 0.0:
            projx = p0[0]; projy = p0[1]; projz = p0[2]
        else:
            t = (wx*vx + wy*vy + wz*vz) / vv
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            projx = p0[0] + t*vx
            projy = p0[1] + t*vy
            projz = p0[2] + t*vz
        dx = sc[0]-projx; dy = sc[1]-projy; dz = sc[2]-projz
        dcenter = (dx*dx + dy*dy + dz*dz)**0.5
        surf = dcenter - (sr + cr)
        if surf < 0.0:
            return 0.0
        return surf

# ---------- unified API: call numba versions if available ----------

def _seg_seg_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    if NUMBA_AVAILABLE:
        return float(_seg_seg_distance_numba(p1, q1, p2, q2))
    return _seg_seg_distance_py(p1, q1, p2, q2)


def cylinder_surface_distance(c1: Cylinder, c2: Cylinder) -> float:
    if NUMBA_AVAILABLE:
        return float(cylinder_surface_distance_numba(c1.p0, c1.p1, c1.r, c2.p0, c2.p1, c2.r))
    return cylinder_surface_distance_py(c1, c2)


def sphere_sphere_surface_distance(s1: Sphere, s2: Sphere) -> float:
    if NUMBA_AVAILABLE:
        return float(sphere_sphere_surface_distance_numba(s1.c, s1.r, s2.c, s2.r))
    return sphere_sphere_surface_distance_py(s1, s2)


def sphere_cylinder_surface_distance(s: Sphere, c: Cylinder) -> float:
    if NUMBA_AVAILABLE:
        return float(sphere_cylinder_surface_distance_numba(s.c, s.r, c.p0, c.p1, c.r))
    return sphere_cylinder_surface_distance_py(s, c)


def segment_plane_distance_to_x_plane(p0: np.ndarray, p1: np.ndarray, x_plane: float) -> float:
    x0 = p0[0]
    x1 = p1[0]
    if (x0 - x_plane) * (x1 - x_plane) <= 0:
        return 0.0
    else:
        return min(abs(x0 - x_plane), abs(x1 - x_plane))
