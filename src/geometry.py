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


# 在 geometry.py 末尾添加

class SphericalCap:
    """
    球冠：一个球被一个平面切割后产生的部分。
    
    表示方式：
        - 原始球心 center (3,)
        - 原始球半径 r
        - 切割平面法向量 n (单位向量，指向保留侧)
        - 切割平面到球心的距离 d（带符号）
        - id: 原始球ID
    
    球冠的几何含义：
        满足 (x - center) · n >= d 的点，且 |x - center| <= r
    
    判断规则（重要）：
        - 完整球（无截断）：d <= -r（整个球在保留侧，没有切割）
        - 真球冠（部分截断）：-r < d < r（切割平面穿过球体）
        - 空球冠（完全被切除）：d >= r（整个球在被切除侧）
    """
    def __init__(self, center: np.ndarray, r: float, n: np.ndarray, d: float, id: int = None):
        self.c = np.asarray(center, dtype=float)
        self.r = float(r)
        norm_n = np.linalg.norm(n)
        if norm_n > 0:
            self.n = np.asarray(n, dtype=float) / norm_n
        else:
            self.n = np.asarray(n, dtype=float)
        self.d = float(d)
        self.id = id
        
    def center(self) -> np.ndarray:
        return self.c
    
    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.c - self.r
        hi = self.c + self.r
        return lo, hi
    
    def is_full_sphere(self) -> bool:
        """球冠为完整球：整个球在切割平面保留侧（d <= -r）"""
        return self.d <= -self.r
    
    def is_empty(self) -> bool:
        """球冠为空：整个球在切割平面被切除侧（d >= r）"""
        return self.d >= self.r
    
    def is_true_cap(self) -> bool:
        """真正的球冠：切割平面穿过球体（-r < d < r）"""
        return -self.r < self.d < self.r
    
    def cap_center(self) -> np.ndarray:
        """球冠底面圆的圆心（在切割平面上）"""
        return self.c + self.d * self.n
    
    def cap_radius(self) -> float:
        """球冠底面圆的半径"""
        if not self.is_true_cap():
            return 0.0
        return np.sqrt(max(0.0, self.r**2 - self.d**2))
    
    def is_point_in_cap(self, p: np.ndarray) -> bool:
        """检查点 p 是否在球冠内"""
        # 检查是否在球内
        if np.linalg.norm(p - self.c) > self.r + 1e-12:
            return False
        # 检查是否在切割平面保留侧
        if np.dot(p - self.c, self.n) < self.d - 1e-12:
            return False
        return True


# ============================================================
# 精确距离计算函数
# ============================================================

def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点到线段的最短距离"""
    v = b - a
    vv = np.dot(v, v)
    if vv == 0:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, v) / vv
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * v
    return np.linalg.norm(p - proj)


def _circle_circle_distance(c1: np.ndarray, r1: float, n1: np.ndarray,
                             c2: np.ndarray, r2: float, n2: np.ndarray) -> float:
    """
    两个圆之间的最短距离（在三维空间中）。
    圆由圆心 c、半径 r、法向量 n 定义。
    """
    # 两个圆心之间的距离
    dc = np.linalg.norm(c1 - c2)
    
    # 如果两个圆共面且圆心距为0，返回 |r1 - r2|
    # 检查法向量是否平行
    n1_n2 = np.dot(n1, n2)
    if abs(abs(n1_n2) - 1.0) < 1e-12:
        # 平行或反平行
        if dc < 1e-12:
            return abs(r1 - r2)
        # 否则，投影到平面上
        # 对于平行圆，最短距离是圆心距减去半径之和（如果圆在投影方向重叠）
        # 更精确：两个圆在垂直于法向量方向上的距离
        # 对于两个平行圆，最短距离 = max(0, sqrt(dc^2 - (r1 - r2)^2) - (r1 + r2))
        # 这实际上是两个圆盘之间的最短距离
        pass
    
    # 一般情况：两个圆在三维空间中的最短距离
    # 使用数值优化或解析方法
    # 对于本问题，使用迭代方法
    best_dist = np.inf
    
    # 采样方法：在每个圆上采样多个点，取最小距离
    n_samples = 36
    for i in range(n_samples):
        theta1 = 2 * np.pi * i / n_samples
        # 构建圆上的点
        # 需要两个正交向量垂直于 n1
        if abs(n1[0]) < 0.9:
            u1 = np.cross(n1, np.array([1., 0., 0.]))
        else:
            u1 = np.cross(n1, np.array([0., 1., 0.]))
        u1 = u1 / np.linalg.norm(u1)
        v1 = np.cross(n1, u1)
        p1 = c1 + r1 * (np.cos(theta1) * u1 + np.sin(theta1) * v1)
        
        for j in range(n_samples):
            theta2 = 2 * np.pi * j / n_samples
            if abs(n2[0]) < 0.9:
                u2 = np.cross(n2, np.array([1., 0., 0.]))
            else:
                u2 = np.cross(n2, np.array([0., 1., 0.]))
            u2 = u2 / np.linalg.norm(u2)
            v2 = np.cross(n2, u2)
            p2 = c2 + r2 * (np.cos(theta2) * u2 + np.sin(theta2) * v2)
            
            dist = np.linalg.norm(p1 - p2)
            if dist < best_dist:
                best_dist = dist
    
    return best_dist


def _circle_sphere_distance(circle_center: np.ndarray, circle_radius: float, 
                             circle_normal: np.ndarray,
                             sphere_center: np.ndarray, sphere_radius: float) -> float:
    """
    圆到球面的最短距离。
    圆由圆心 cc、半径 cr、法向量 n 定义。
    球由球心 sc、半径 sr 定义。
    """
    # 球心到圆所在平面的距离
    d_plane = np.dot(sphere_center - circle_center, circle_normal)
    # 球心在平面上的投影
    proj = sphere_center - d_plane * circle_normal
    # 投影点到圆心的距离
    d_proj = np.linalg.norm(proj - circle_center)
    
    if d_proj <= circle_radius:
        # 球心投影在圆内，最短距离是球心到平面的距离减去球半径
        return max(0.0, abs(d_plane) - sphere_radius)
    else:
        # 球心投影在圆外，最短距离是球心到圆边界的距离减去球半径
        d_circle = np.linalg.norm(proj - circle_center) - circle_radius
        # 到圆的距离是 sqrt(d_plane^2 + d_circle^2)
        dist_to_circle = np.sqrt(d_plane**2 + d_circle**2)
        return max(0.0, dist_to_circle - sphere_radius)


def _circle_cylinder_distance(circle_center: np.ndarray, circle_radius: float,
                               circle_normal: np.ndarray,
                               cyl_p0: np.ndarray, cyl_p1: np.ndarray, cyl_r: float) -> float:
    """
    圆到圆柱面的最短距离。
    圆由圆心 cc、半径 cr、法向量 n 定义。
    圆柱由轴端点 p0, p1 和半径 cr 定义。
    """
    # 采样方法：在圆上采样点，计算到圆柱的最短距离
    n_samples = 48
    best_dist = np.inf
    
    # 构建圆上的正交向量
    if abs(circle_normal[0]) < 0.9:
        u = np.cross(circle_normal, np.array([1., 0., 0.]))
    else:
        u = np.cross(circle_normal, np.array([0., 1., 0.]))
    u = u / np.linalg.norm(u)
    v = np.cross(circle_normal, u)
    
    for i in range(n_samples):
        theta = 2 * np.pi * i / n_samples
        p = circle_center + circle_radius * (np.cos(theta) * u + np.sin(theta) * v)
        
        # 点到圆柱轴线的距离
        dist_to_axis = _point_segment_distance(p, cyl_p0, cyl_p1)
        # 到圆柱表面的距离
        surf_dist = max(0.0, dist_to_axis - cyl_r)
        if surf_dist < best_dist:
            best_dist = surf_dist
    
    return best_dist


def sphericalcap_sphere_surface_distance(cap: SphericalCap, sph: Sphere) -> float:
    """
    球冠与完整球之间的精确最短距离。
    """
    # 1. 检查完整球之间的最短距离点是否在球冠内
    d_center = np.linalg.norm(cap.c - sph.c)
    if d_center > 0:
        dir_vec = (sph.c - cap.c) / d_center
        # 球冠上的最近点（完整球上）
        p_on_cap = cap.c + cap.r * dir_vec
        # 球上的最近点
        p_on_sphere = sph.c - sph.r * dir_vec
        
        # 检查 p_on_cap 是否在球冠内
        if cap.is_point_in_cap(p_on_cap):
            # 完整球距离就是精确距离
            return max(0.0, d_center - cap.r - sph.r)
    
    # 2. 完整球距离不成立，需要检查球冠边界圆到球的距离
    cap_circle_center = cap.cap_center()
    cap_circle_radius = cap.cap_radius()
    
    if cap_circle_radius > 0:
        return _circle_sphere_distance(
            cap_circle_center, cap_circle_radius, cap.n,
            sph.c, sph.r
        )
    else:
        # 球冠退化为一个点（d = r 或 d = -r）
        # 实际上 d = -r 时球冠为空，d = r 时球冠退化为一个点
        if cap.d >= cap.r - 1e-12:
            # 退化为点
            return max(0.0, np.linalg.norm(cap.c + cap.r * cap.n - sph.c) - sph.r)
        else:
            # 空球冠
            return np.inf


def sphericalcap_cylinder_surface_distance(cap: SphericalCap, cyl: Cylinder) -> float:
    """
    球冠与圆柱之间的精确最短距离。
    """
    # 1. 检查完整球到圆柱的最短距离点是否在球冠内
    # 球心到圆柱轴线的最近点
    v = cyl.p1 - cyl.p0
    vv = np.dot(v, v)
    if vv == 0:
        proj_on_axis = cyl.p0
    else:
        t = np.dot(cap.c - cyl.p0, v) / vv
        t_clamped = max(0.0, min(1.0, t))
        proj_on_axis = cyl.p0 + t_clamped * v
    
    # 球心到轴线的距离
    d_axis = np.linalg.norm(cap.c - proj_on_axis)
    
    if d_axis > 0:
        # 球上最近点（指向轴线方向）
        dir_to_axis = (cap.c - proj_on_axis) / d_axis
        p_on_sphere = cap.c - cap.r * dir_to_axis
        
        # 检查是否在球冠内
        if cap.is_point_in_cap(p_on_sphere):
            # 完整球到圆柱的距离
            return max(0.0, d_axis - cap.r - cyl.r)
    
    # 2. 检查球冠边界圆到圆柱的距离
    cap_circle_center = cap.cap_center()
    cap_circle_radius = cap.cap_radius()
    
    if cap_circle_radius > 0:
        return _circle_cylinder_distance(
            cap_circle_center, cap_circle_radius, cap.n,
            cyl.p0, cyl.p1, cyl.r
        )
    else:
        # 球冠退化为点
        if cap.d >= cap.r - 1e-12:
            p = cap.c + cap.r * cap.n
            dist_to_axis = _point_segment_distance(p, cyl.p0, cyl.p1)
            return max(0.0, dist_to_axis - cyl.r)
        else:
            return np.inf


def sphericalcap_sphericalcap_surface_distance(cap1: SphericalCap, cap2: SphericalCap) -> float:
    """
    两个球冠之间的精确最短距离。
    """
    # 1. 检查两个完整球的最短距离点是否分别在两个球冠内
    d_center = np.linalg.norm(cap1.c - cap2.c)
    if d_center > 0:
        dir_vec = (cap2.c - cap1.c) / d_center
        p1 = cap1.c + cap1.r * dir_vec  # cap1 上的最近点
        p2 = cap2.c - cap2.r * dir_vec  # cap2 上的最近点
        
        in1 = cap1.is_point_in_cap(p1)
        in2 = cap2.is_point_in_cap(p2)
        
        if in1 and in2:
            return max(0.0, d_center - cap1.r - cap2.r)
    
    # 2. 获取两个球冠的边界圆参数
    cap1_circle_center = cap1.cap_center()
    cap1_circle_radius = cap1.cap_radius()
    cap2_circle_center = cap2.cap_center()
    cap2_circle_radius = cap2.cap_radius()
    
    best_dist = np.inf
    
    # 3. 采样方法：在 cap1 边界圆上采样点，计算到 cap2 的距离
    n_samples = 48
    
    # 构建 cap1 圆上的正交向量
    if abs(cap1.n[0]) < 0.9:
        u1 = np.cross(cap1.n, np.array([1., 0., 0.]))
    else:
        u1 = np.cross(cap1.n, np.array([0., 1., 0.]))
    u1 = u1 / np.linalg.norm(u1)
    v1 = np.cross(cap1.n, u1)
    
    for i in range(n_samples):
        theta = 2 * np.pi * i / n_samples
        p = cap1_circle_center + cap1_circle_radius * (np.cos(theta) * u1 + np.sin(theta) * v1)
        
        # 计算 p 到 cap2 的距离
        d_to_cap2 = _point_to_sphericalcap_distance(p, cap2)
        if d_to_cap2 < best_dist:
            best_dist = d_to_cap2
    
    # 4. 反过来采样 cap2 边界圆上的点（确保对称性）
    if cap2_circle_radius > 0:
        if abs(cap2.n[0]) < 0.9:
            u2 = np.cross(cap2.n, np.array([1., 0., 0.]))
        else:
            u2 = np.cross(cap2.n, np.array([0., 1., 0.]))
        u2 = u2 / np.linalg.norm(u2)
        v2 = np.cross(cap2.n, u2)
        
        for i in range(n_samples):
            theta = 2 * np.pi * i / n_samples
            p = cap2_circle_center + cap2_circle_radius * (np.cos(theta) * u2 + np.sin(theta) * v2)
            
            d_to_cap1 = _point_to_sphericalcap_distance(p, cap1)
            if d_to_cap1 < best_dist:
                best_dist = d_to_cap1
    
    return best_dist if best_dist < np.inf else 0.0


def _point_to_sphericalcap_distance(p: np.ndarray, cap: SphericalCap) -> float:
    """
    点到球冠的最短距离。
    """
    # 检查点是否在球冠内
    if cap.is_point_in_cap(p):
        return 0.0
    
    # 计算点到完整球的距离
    dist_to_sphere = max(0.0, np.linalg.norm(p - cap.c) - cap.r)
    
    # 计算点到切割平面的距离（在球冠侧）
    # 如果点在球冠的"背面"（即被切除的一侧），需要特殊处理
    proj_on_n = np.dot(p - cap.c, cap.n)
    
    if proj_on_n < cap.d:
        # 点在切割平面的被切除侧
        # 到球冠的最短距离是到切割平面与球的交线（边界圆）的距离
        cap_circle_center = cap.cap_center()
        cap_circle_radius = cap.cap_radius()
        
        if cap_circle_radius > 0:
            # 计算点到边界圆的距离
            # 先投影到切割平面
            proj_on_plane = p - (proj_on_n - cap.d) * cap.n
            # 投影点到圆心的距离
            dist_to_center = np.linalg.norm(proj_on_plane - cap_circle_center)
            if dist_to_center <= cap_circle_radius:
                # 投影在圆内，到球冠的距离就是到切割平面的距离
                return max(0.0, cap.d - proj_on_n)
            else:
                # 投影在圆外，到边界圆的距离
                dist_to_circle = dist_to_center - cap_circle_radius
                # 到球冠的距离是到边界圆的距离（在三维空间中）
                # 注意：需要同时考虑垂直于切割平面的方向
                return np.sqrt((cap.d - proj_on_n)**2 + dist_to_circle**2)
        else:
            # 退化为点
            return max(0.0, np.linalg.norm(p - (cap.c + cap.r * cap.n)))
    
    # 点在球冠的保留侧，但不在球内
    # 到球冠的最短距离就是到完整球的距离
    return dist_to_sphere