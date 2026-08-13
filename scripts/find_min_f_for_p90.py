#!/usr/bin/env python3
"""
scripts/find_min_f_for_p90.py

Find minimal volume fraction f such that conduction probability p(f) >= target_prob (default 0.9),
using Monte Carlo trials and Wilson confidence intervals.

Usage (example):
    python scripts/find_min_f_for_p90.py --target 0.9 --tol 0.0005 --fmax 0.2 --initial_trials 200 --max_trials 2000 --processes 4

Notes:
- Run from repository root (so src/ is importable).
- On Windows, run in terminal (not notebook) to use multiprocessing safely.
"""

import os
import sys
import argparse
import time
import math
from multiprocessing import Pool
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd

# add repo root to path (script in repo/scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# import project modules
try:
    from src.geometry import Cylinder
    from src.truncation import split_cylinder_by_box
    from src.connectivity import build_connectivity
except Exception as e:
    print("ERROR importing src modules:", e)
    raise

# ---------------- core utilities ----------------
def volume_cylinder(rad, length):
    return math.pi * (rad ** 2) * length

def compute_N_from_f(f, Lbox, rad, length):
    V_box = Lbox ** 3
    V_cyl = volume_cylinder(rad, length)
    if V_cyl <= 0:
        raise ValueError("Cylinder volume must be positive")
    return max(0, int(round((f * V_box) / V_cyl)))

def sample_random_cylinders(N, Lbox, rad, length, seed=None):
    if N <= 0:
        return []
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

# worker for a single trial
def _run_single_trial(args):
    N, Lbox, rad, length, thresh, seed = args
    cyls = sample_random_cylinders(N, Lbox, rad, length, seed=seed)
    segments = []
    for c in cyls:
        segs = split_cylinder_by_box(c, Lbox)
        segments.extend(segs)
    connected, _ = build_connectivity(segments, Lbox, thresh)
    return 1 if connected else 0

def estimate_p(f, trials, L, r, h, thresh, seed_base=0, processes=1):
    N = compute_N_from_f(f, L, r, h)
    if trials <= 0 or N == 0:
        return {'f': f, 'N': N, 'trials': trials, 'successes': 0, 'p_hat': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
    args_list = []
    for t in range(trials):
        seed = int(seed_base + t + (int(abs(f)*1e9) % 1000000))
        args_list.append((N, L, r, h, thresh, seed))
    if processes == 1:
        results = [_run_single_trial(a) for a in args_list]
    else:
        with Pool(processes) as pool:
            results = pool.map(_run_single_trial, args_list)
    successes = sum(results)
    p_hat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)
    return {'f': f, 'N': N, 'trials': trials, 'successes': successes, 'p_hat': p_hat, 'ci_low': ci_low, 'ci_high': ci_high}

# ---------------- search logic ----------------
def find_bracket_for_target(target, f_start, f_max, L, r, h, thresh, initial_trials, processes, seed_base):
    """
    Expand f upward (multiplicative or additive) to find f_high where p_hat >= target (coarsely).
    Returns (f_low, f_high, res_low, res_high) where res_* are estimate dicts.
    """
    f_low = max(1e-8, f_start)
    res_low = estimate_p(f_low, initial_trials, L, r, h, thresh, seed_base=seed_base, processes=processes)
    if res_low['ci_low'] >= target:
        return (0.0, f_low, None, res_low)  # already satisfied at tiny f
    # progressive expansion: try linear steps first, then geometric if needed
    f = f_low
    # try coarse linear increments up to f_max
    steps = [f + i*(f_max - f) / 10.0 for i in range(1,11)]
    for f_try in steps:
        if f_try > f_max:
            break
        res = estimate_p(f_try, max(50, int(initial_trials/2)), L, r, h, thresh, seed_base=seed_base, processes=processes)
        if res['ci_low'] >= target or res['p_hat'] >= target:
            return (f, f_try, res_low, res)
    # if not found, try geometric growth starting from f (double)
    f_try = max(steps[-1] if steps else f, f*2.0)
    while f_try <= f_max:
        res = estimate_p(f_try, max(50, int(initial_trials/2)), L, r, h, thresh, seed_base=seed_base, processes=processes)
        if res['ci_low'] >= target or res['p_hat'] >= target:
            return (f, f_try, res_low, res)
        f = f_try
        res_low = res
        f_try = min(f_try*2.0, f_max)
        if abs(f_try - f) < 1e-12:
            break
    # not found
    return (f, f_max, res_low, None)

def binary_search_min_f(target, f_lo, f_hi, L, r, h, thresh, initial_trials, max_trials, processes, tol, seed_base):
    """
    Binary search in [f_lo, f_hi] to find minimal f so that lower CI >= target (or p_hat >= target).
    Uses adaptive trials: at each mid, run with increasing trials if result is inconclusive,
    up to max_trials.
    """
    left = f_lo
    right = f_hi
    best = None
    history = []

    while right - left > tol:
        mid = 0.5 * (left + right)
        trials = initial_trials
        res = estimate_p(mid, trials, L, r, h, thresh, seed_base=seed_base, processes=processes)
        history.append(res)
        # If lower CI already >= target -> we can move left
        if res['ci_low'] >= target:
            best = res
            right = mid
            continue
        # If upper CI < target -> mid too small
        if res['ci_high'] < target:
            left = mid
            continue
        # inconclusive: increase trials progressively
        while trials < max_trials:
            trials = min(trials * 2, max_trials)
            res = estimate_p(mid, trials, L, r, h, thresh, seed_base=seed_base, processes=processes)
            history.append(res)
            if res['ci_low'] >= target:
                best = res
                right = mid
                break
            if res['ci_high'] < target:
                left = mid
                break
        else:
            # reached max_trials and still inconclusive: decide by p_hat
            if res['p_hat'] >= target:
                best = res
                right = mid
            else:
                left = mid
        # safety: if left converges to 0 and best None -> break to avoid infinite loop
        if right - left <= tol:
            break

    return best, history, (left, right)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Find minimal f so that p(f) >= target_prob (approx).")
    p.add_argument('--target', type=float, default=0.9, help='target probability (default 0.9)')
    p.add_argument('--fstart', type=float, default=0.001, help='starting f to search from (default 0.001)')
    p.add_argument('--fmax', type=float, default=0.2, help='maximum f to consider (default 0.2)')
    p.add_argument('--initial_trials', type=int, default=200, help='initial trials per evaluation (default 200)')
    p.add_argument('--max_trials', type=int, default=2000, help='max trials per evaluation (default 2000)')
    p.add_argument('--processes', type=int, default=4, help='worker processes (default 4)')
    p.add_argument('--tol', type=float, default=5e-4, help='tolerance for f (default 5e-4)')
    p.add_argument('--L', type=float, default=10000.0, help='box size nm (default 10000)')
    p.add_argument('--r', type=float, default=30.0, help='cylinder radius nm (default 30)')
    p.add_argument('--h', type=float, default=5000.0, help='cylinder length nm (default 5000)')
    p.add_argument('--thresh', type=float, default=1.8, help='surface-to-surface threshold nm (default 1.8)')
    p.add_argument('--seed', type=int, default=20260812, help='seed base (default 20260812)')
    p.add_argument('--out', type=str, default=os.path.join('results','outputs','find_min_f_p90.csv'), help='output CSV for logging')
    return p.parse_args()

def safe_save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.replace('.csv', f'_{int(time.time())}.csv')
        df.to_csv(alt, index=False)
        return alt

def main():
    args = parse_args()
    print("Find minimal f for target p =", args.target)
    print("Params: L={}, r={}, h={}, thresh={}".format(args.L, args.r, args.h, args.thresh))
    # set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    # Step 1: bracket f
    print("Bracketing ...")
    f_lo, f_hi, res_lo, res_hi = find_bracket_for_target(args.target, args.fstart, args.fmax, args.L, args.r, args.h, args.thresh, args.initial_trials, args.processes, args.seed)
    print(f"Bracket result: f_lo={f_lo}, f_hi={f_hi}")
    if res_hi is None:
        print("Failed to find upper bracket within fmax. Try increasing fmax or inspect parameter choices.")
        # Still save res_lo
        records = []
        if res_lo is not None:
            records.append(res_lo)
        df = pd.DataFrame(records)
        saved = safe_save(df, args.out)
        print("Saved partial results to", saved)
        return

    print("Refining with binary search ...")
    best, history, (left, right) = binary_search_min_f(args.target, f_lo, f_hi, args.L, args.r, args.h, args.thresh, args.initial_trials, args.max_trials, args.processes, args.tol, args.seed)

    # Consolidate history to DataFrame
    df_hist = pd.DataFrame(history)
    out_path = args.out
    saved_path = safe_save(df_hist, out_path)
    print("Saved search history to", saved_path)

    if best is None:
        print("Search finished but did not find a configuration with sufficient confidence (best None).")
        print("Last bracket:", left, right)
        print("You may increase max_trials or fmax.")
    else:
        print("Found candidate minimal f (approx):", best['f'])
        print("Details:", best)
        print("Score interval around final bracket: [{:.6g}, {:.6g}]".format(left, right))
        # Suggest re-evaluating best['f'] with higher trials for final result
        print("Suggested next step: re-evaluate at f = {:.6g} with higher trials (e.g., 2000-5000) to confirm.".format(best['f']))

    # save summary
    summary = {
        'target': args.target,
        'found_f': best['f'] if best is not None else None,
        'p_hat': best['p_hat'] if best is not None else None,
        'ci_low': best['ci_low'] if best is not None else None,
        'ci_high': best['ci_high'] if best is not None else None,
        'left_bracket': left, 'right_bracket': right,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    df_sum = pd.DataFrame([summary])
    sum_path = out_path.replace('.csv', '_summary.csv')
    sum_saved = safe_save(df_sum, sum_path)
    print("Saved summary to", sum_saved)

if __name__ == '__main__':
    main()