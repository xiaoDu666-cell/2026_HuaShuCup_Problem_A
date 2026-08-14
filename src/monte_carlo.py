from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .geometry import Cylinder, Sphere
from .truncation import split_cylinder_by_box, split_sphere_by_box
from .connectivity import build_connectivity

def random_unit_vector():
    v = np.random.normal(size=3)
    n = np.linalg.norm(v)
    if n == 0:
        return np.array([1.0, 0.0, 0.0])
    return v / n

def sample_random_cylinders(N: int, L: float, r: float, h: float, start_id: int = 0, rng=None) -> List[Cylinder]:
    if rng is None:
        rng = np.random.default_rng()
    half = L/2.0
    cyls = []
    for i in range(N):
        center = rng.uniform(-half, half, size=3)
        direction = random_unit_vector()
        halfvec = 0.5 * h * direction
        p0 = center - halfvec
        p1 = center + halfvec
        cyls.append(Cylinder(p0, p1, r, id=start_id + i))
    return cyls

def sample_random_spheres(N: int, L: float, radius: float, start_id: int = 0, rng=None) -> List[Sphere]:
    if rng is None:
        rng = np.random.default_rng()
    half = L/2.0
    sphs = []
    for i in range(N):
        center = rng.uniform(-half, half, size=3)
        sphs.append(Sphere(center, radius, id=start_id + i))
    return sphs

def run_single_trial_mixed(N_A: int, N_B: int, L: float, r_A: float, h_A: float, r_B: float, thresh: float, seed: int = None):
    rng = np.random.default_rng(seed)
    cyls = sample_random_cylinders(N_A, L, r_A, h_A, start_id=0, rng=rng)
    sphs = sample_random_spheres(N_B, L, r_B, start_id=N_A, rng=rng)
    particles = []
    for c in cyls:
        particles.extend(split_cylinder_by_box(c, L))
    for s in sphs:
        particles.extend(split_sphere_by_box(s, L))
    connected, uf = build_connectivity(particles, L, thresh)
    return connected, particles, uf

def run_trials_mixed(n_trials: int, N_A: int, N_B: int, L: float, r_A: float, h_A: float, r_B: float, thresh: float, seed_base: int = 0):
    successes = 0
    for t in range(n_trials):
        seed = int(seed_base + t + (N_A * 1000003) + (N_B * 10007))
        connected, _, _ = run_single_trial_mixed(N_A, N_B, L, r_A, h_A, r_B, thresh, seed=seed)
        if connected:
            successes += 1
    return successes / n_trials if n_trials > 0 else 0.0