#边界截断规则的实现（解析分割方法）
# truncation.py
# Functions to split/truncate cylinders by a cubic box with mapping rule described in the problem.

from __future__ import annotations
import numpy as np
from typing import List
from .geometry import Cylinder

def _plane_ts_for_axis(A_coord, B_coord, plane_pos):
    """Return t where segment coordinate equals plane_pos: t = (plane - A) / (B - A)
    Only returns t if denominator != 0.
    """
    denom = B_coord - A_coord
    if abs(denom) < 1e-15:
        return []
    t = (plane_pos - A_coord) / denom
    return [t]

def split_cylinder_by_box(cyl: Cylinder, L: float) -> List[Cylinder]:
    """Split a cylinder axis into segments at box faces x=+-L/2,y=...,z=..., then map each segment back into box
    according to the rule: if a segment lies outside in a given direction, translate it by +/-L so it sits inside.
    Returns a list of Cylinder segments (each with same radius and same original id).
    """
    A = cyl.p0
    B = cyl.p1
    planes = []
    half = L / 2.0
    # Generate candidate t values for intersections with the 6 faces
    ts = [0.0, 1.0]
    for dim in range(3):
        a_coord = A[dim]
        b_coord = B[dim]
        for face in (-half, half):
            if abs(b_coord - a_coord) < 1e-15:
                continue
            t = (face - a_coord) / (b_coord - a_coord)
            if 0.0 < t < 1.0:
                ts.append(t)
    ts = sorted(set(ts))
    segments = []
    for i in range(len(ts) - 1):
        t0 = ts[i]
        t1 = ts[i + 1]
        p0 = A + (B - A) * t0
        p1 = A + (B - A) * t1
        midpoint = 0.5 * (p0 + p1)
        # compute shift per axis to bring midpoint into [-half,half]
        shift = np.zeros(3)
        for d in range(3):
            if midpoint[d] > half:
                shift[d] = -L
            elif midpoint[d] < -half:
                shift[d] = L
            else:
                shift[d] = 0.0
        p0_shift = p0 + shift
        p1_shift = p1 + shift
        # clamp tiny numerical out-of-bound values
        p0_shift = np.minimum(np.maximum(p0_shift, -half), half)
        p1_shift = np.minimum(np.maximum(p1_shift, -half), half)
        segments.append(Cylinder(p0_shift, p1_shift, cyl.r, id=cyl.id))
    return segments