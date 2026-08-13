#!/usr/bin/env python3
"""
scripts/fine_scan_f_values.py

Run fine-grained Monte Carlo at specified f values and find first f with p_hat >= threshold.

Example:
  python scripts/fine_scan_f_values.py \
    --flist 0.0005,0.0006,0.0007,0.0008,0.0009,0.0010 \
    --trials 2000 --processes 4 --out results/outputs/fine_scan_f_values.csv

Notes:
- Run from repository root so src/ modules are importable.
- In Jupyter set --processes 1 to avoid multiprocessing issues.
"""
import os
import sys
import argparse
import time
import math
import pandas as pd
import numpy as np
from multiprocessing import Pool

# ensure repo root in path
repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# import project modules
from src.geometry import Cylinder
from src.truncation import split_cylinder_by_box
from src.connectivity import build_connectivity

def volume_cylinder(rad, length):
    return math.pi * (rad ** 2) * length

def compute_N_from_f(f, Lbox, rad, length):
    V_box = Lbox**3
    V_cyl = volume_cylinder(rad, length)
    if V_cyl <= 0:
        raise ValueError("Cylinder volume must be positive")
    return max(0, int(round((f * V_box) / V_cyl)))

def sample_random_cylinders(N, Lbox, rad, length, seed=None):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-Lbox/2, Lbox/2, size=(N, 3))
    dirs = rng.normal(size=(N, 3))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = dirs / norms
    half = length / 2.0
    cyls = []
    for i in range(N):
        c = centers[i]
        d = dirs[i]
        p0 = c - d * half
        p1 = c + d * half
        cyls.append(Cylinder(p0, p1, rad, id=i))
    return cyls

def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + (z**2) / n
    centre = phat + (z**2) / (2 * n)
    adj = z * math.sqrt((phat*(1-phat) + (z**2) / (4*n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

# single-trial worker
def _run_single_trial(args):
    N, Lbox, rad, length, thresh, seed = args
    cyls = sample_random_cylinders(N, Lbox, rad, length, seed=seed)
    segments = []
    for c in cyls:
        segments.extend(split_cylinder_by_box(c, Lbox))
    connected, _ = build_connectivity(segments, Lbox, thresh)
    return 1 if connected else 0

def estimate_p(f, trials, L, r, h, thresh, seed_base=0, processes=1):
    N = compute_N_from_f(f, L, r, h)
    if trials <= 0 or N == 0:
        return {'f': f, 'N': N, 'trials': trials, 'successes': 0, 'p_hat': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'elapsed_s': 0.0}
    args_list = []
    for t in range(trials):
        seed = int(seed_base + t + (int(abs(f) * 1e9) % 1000000))
        args_list.append((N, L, r, h, thresh, seed))
    t0 = time.time()
    if processes == 1:
        results = [_run_single_trial(a) for a in args_list]
    else:
        with Pool(processes) as pool:
            results = pool.map(_run_single_trial, args_list)
    successes = int(sum(results))
    phat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)
    elapsed = time.time() - t0
    return {'f': f, 'N': N, 'trials': trials, 'successes': successes, 'p_hat': phat, 'ci_low': ci_low, 'ci_high': ci_high, 'elapsed_s': elapsed}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--flist', type=str, default="0.0005,0.0006,0.0007,0.0008,0.0009,0.0010", help='comma list of f values (fractions)')
    p.add_argument('--trials', type=int, default=2000, help='trials per f (default 2000)')
    p.add_argument('--processes', type=int, default=4, help='worker processes (default 4)')
    p.add_argument('--L', type=float, default=10000.0)
    p.add_argument('--r', type=float, default=30.0)
    p.add_argument('--h', type=float, default=5000.0)
    p.add_argument('--thresh', type=float, default=1.8)
    p.add_argument('--seed', type=int, default=20260812)
    p.add_argument('--out', type=str, default=os.path.join('results','outputs','fine_scan_f_values.csv'))
    p.add_argument('--require-ci-low', action='store_true', help='require ci_low >= threshold instead of p_hat >= threshold')
    p.add_argument('--threshold', type=float, default=0.90, help='target probability threshold')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f_list = [float(x) for x in args.flist.split(',') if x.strip()!= '']
    f_list = sorted(f_list)
    results = []
    found = False
    for f in f_list:
        print(f"\nEvaluating f={f:.6g} with trials={args.trials} ...")
        res = estimate_p(f, args.trials, args.L, args.r, args.h, args.thresh, seed_base=args.seed, processes=args.processes)
        results.append(res)
        df = pd.DataFrame(results)
        df.to_csv(args.out, index=False)
        print(f"  f={f}, N={res['N']}, successes={res['successes']}/{res['trials']}, p_hat={res['p_hat']:.4f}, ci=({res['ci_low']:.4f},{res['ci_high']:.4f}), time={res['elapsed_s']:.1f}s")
        # decide
        if args.require_ci_low:
            cond = (res['ci_low'] >= args.threshold)
        else:
            cond = (res['p_hat'] >= args.threshold)
        if cond:
            print(f"\n==> Found first f meeting requirement at f = {f} (stopping scan).")
            found = True
            break
    if not found:
        print("\nScan finished: no f in the list reached the threshold under the chosen decision rule.")
    print(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()