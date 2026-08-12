#蒙特卡洛模拟的通用框架（随机生成、并行执行）
# monte_carlo.py
# Random generation of cylinders and Monte Carlo trial runner.

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .geometry import Cylinder
from .truncation import split_cylinder_by_box
from .connectivity import build_connectivity

def random_unit_vector():
    v = np.random.normal(size=3)
    n = np.linalg.norm(v)
    return v / n

def sample_random_cylinders(N: int, L: float, r: float, h: float, start_id: int = 0) -> List[Cylinder]:
    """Sample N random cylinders (center uniform in box, orientation uniform on sphere)."""
    half = L/2.0
    cyls = []
    for i in range(N):
        center = np.random.uniform(-half, half, size=3)
        direction = random_unit_vector()
        halfvec = 0.5 * h * direction
        p0 = center - halfvec
        p1 = center + halfvec
        cyls.append(Cylinder(p0, p1, r, id=start_id + i))
    return cyls

def run_single_trial_from_centers(cyls: List[Cylinder], L: float, thresh: float):
    """Given list of sampled cylinders (full), apply truncation and check connectivity."""
    segments = []
    for c in cyls:
        segs = split_cylinder_by_box(c, L)
        # segments already carry same id
        segments.extend(segs)
    connected, uf = build_connectivity(segments, L, thresh)
    return connected

def run_trials(n_trials: int, N: int, L: float, r: float, h: float, thresh: float, parallel: bool = False) -> float:
    """Run Monte Carlo trials; returns estimated connectivity probability."""
    successes = 0
    for t in range(n_trials):
        cyls = sample_random_cylinders(N, L, r, h, start_id=0)
        if run_single_trial_from_centers(cyls, L, thresh):
            successes += 1
    return successes / n_trials if n_trials > 0 else 0.0