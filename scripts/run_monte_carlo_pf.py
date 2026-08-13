#!/usr/bin/env python3
"""
scripts/run_monte_carlo_pf.py

Monte Carlo estimate of conduction probability p(f) for different volume fractions f.

Usage examples:
    python scripts/run_monte_carlo_pf.py --fmin 0.001 --fmax 0.05 --npoints 12 --trials 200 --processes 4
    python scripts/run_monte_carlo_pf.py --flist 0.001,0.002,0.005,0.01 --trials 500 --processes 8

Notes:
- Designed to be run from repo root so src/ is importable. If not, adjust PYTHONPATH or run from repo root.
- On Windows, run from terminal (not in notebook) to use multiprocessing safely.
"""

import os
import sys
import time
import argparse
import math
import csv
from multiprocessing import Pool
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd

# Try to import tqdm for nicer progress; fallback to simple prints
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Ensure repo root is on sys.path (script placed under repo/scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import project geometry/connectivity implementations
try:
    from src.geometry import Cylinder
    from src.truncation import split_cylinder_by_box
    from src.connectivity import build_connectivity
except Exception as e:
    print("ERROR importing project modules from src/:", e)
    print("Make sure this script is run from the repository root and src/ is importable.")
    raise

# ---------------- helper functions ----------------
def volume_cylinder(rad, length):
    return math.pi * (rad ** 2) * length

def compute_N_from_f(f, Lbox, rad, length):
    V_box = Lbox ** 3
    V_cyl = volume_cylinder(rad, length)
    if V_cyl <= 0:
        raise ValueError("Cylinder volume must be positive")
    return max(0, int(round((f * V_box) / V_cyl)))

def sample_random_cylinders(N, Lbox, rad, length, seed=None):
    """Return list of Cylinder objects (centers uniform in box, directions uniform on sphere)."""
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
    """Wilson score interval for proportion k/n"""
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + (z**2)/n
    centre = phat + (z**2)/(2*n)
    adj = z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

# ---------------- trial-level worker ----------------
def run_single_trial(args):
    """Worker for a single independent trial. Returns 1 if connected, else 0.
    args: (N, L, r, h, thresh, seed)
    """
    N, Lbox, rad, length, thr, seed = args
    cyls = sample_random_cylinders(N, Lbox, rad, length, seed=seed)
    segments = []
    for c in cyls:
        segs = split_cylinder_by_box(c, Lbox)
        segments.extend(segs)
    connected, _uf = build_connectivity(segments, Lbox, thr)
    return 1 if connected else 0

# ---------------- per-f Monte Carlo ----------------
def monte_carlo_for_f(f, trials, L, r, h, thresh, seed_base=0, processes=1, progress=False):
    N = compute_N_from_f(f, L, r, h)
    if trials <= 0 or N == 0:
        return {'f': float(f), 'N': N, 'trials': int(trials), 'successes': 0, 'p_hat': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'elapsed_s': 0.0}
    args_list = []
    for t in range(trials):
        seed = int(seed_base + t + (int(abs(f) * 1e9) % 1000000))
        args_list.append((N, L, r, h, thresh, seed))

    t0 = time.time()
    if processes == 1:
        # sequential
        if progress and _HAS_TQDM:
            results = [run_single_trial(a) for a in tqdm(args_list, desc=f"f={f:.6f}")]
        else:
            results = [run_single_trial(a) for a in args_list]
    else:
        # multiprocessing Pool
        # Note: ensure this function is defined at module level (it is) so Pool can pickle it
        with Pool(processes) as pool:
            if progress and _HAS_TQDM:
                # use imap to provide progress
                results = []
                for res in tqdm(pool.imap(run_single_trial, args_list), total=len(args_list), desc=f"f={f:.6f}"):
                    results.append(res)
            else:
                results = pool.map(run_single_trial, args_list)
    successes = int(sum(results))
    p_hat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)
    elapsed = time.time() - t0
    return {'f': float(f), 'N': int(N), 'trials': int(trials), 'successes': successes, 'p_hat': float(p_hat), 'ci_low': float(ci_low), 'ci_high': float(ci_high), 'elapsed_s': float(elapsed)}

# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo p(f) scanner for conduction probability")
    p.add_argument('--fmin', type=float, default=0.001, help='minimum volume fraction f (default 0.001)')
    p.add_argument('--fmax', type=float, default=0.02, help='maximum volume fraction f (default 0.02)')
    p.add_argument('--npoints', type=int, default=20, help='number of f points between fmin and fmax (default 20)')
    p.add_argument('--flist', type=str, default=None, help='comma-separated list of f values (overrides fmin/fmax/npoints)')
    p.add_argument('--trials', type=int, default=500, help='trials per f (default 500)')
    p.add_argument('--processes', type=int, default=4, help='number of worker processes (default 4)')
    p.add_argument('--seed', type=int, default=20260812, help='seed base (default 20260812)')
    p.add_argument('--L', type=float, default=10000.0, help='box edge length (nm)')
    p.add_argument('--r', type=float, default=30.0, help='cylinder radius (nm)')
    p.add_argument('--h', type=float, default=5000.0, help='cylinder length (nm)')
    p.add_argument('--thresh', type=float, default=1.8, help='surface-to-surface threshold (nm)')
    p.add_argument('--out', type=str, default=os.path.join('results','outputs','monte_carlo_pf.csv'), help='output CSV path')
    p.add_argument('--no-progress', action='store_true', help='disable progress bar (tqdm)')
    return p.parse_args()

def safe_save_df(df, out_path):
    """Try to save df to out_path; on permission error try timestamped alternative."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path
    except PermissionError as e:
        alt = out_path.replace('.csv', f'_{int(time.time())}.csv')
        try:
            df.to_csv(alt, index=False)
            print(f"Warning: permission denied writing {out_path}. Saved to alternative {alt}")
            return alt
        except Exception as e2:
            print(f"ERROR: failed to save to both {out_path} and {alt}: {e2}")
            raise

def main():
    args = parse_args()
    # Compute f list
    if args.flist:
        try:
            f_list = [float(x) for x in args.flist.split(',') if x.strip()!='']
            f_list = sorted(set(f_list))
        except Exception:
            raise ValueError("Unable to parse --flist. Provide comma-separated numeric values.")
    else:
        f_list = list(np.linspace(args.fmin, args.fmax, args.npoints))

    # Basic logging
    print("Monte Carlo p(f) scan")
    print("Parameters: L={}, r={}, h={}, thresh={}, trials={}, processes={}".format(args.L, args.r, args.h, args.thresh, args.trials, args.processes))
    print("f values:", f_list)

    # Try setting start method (Windows)
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    results = []
    out_path = args.out

    # iterate f values
    for f in f_list:
        print(f"\nRunning f = {f:.6f}")
        res = monte_carlo_for_f(f, args.trials, args.L, args.r, args.h, args.thresh, seed_base=args.seed, processes=args.processes, progress=(not args.no_progress and _HAS_TQDM))
        print(f"  N={res['N']}, successes={res['successes']}/{res['trials']}, p_hat={res['p_hat']:.4f}, ci=({res['ci_low']:.4f},{res['ci_high']:.4f}), time={res['elapsed_s']:.1f}s")
        results.append(res)
        # Save intermediate results
        df_tmp = pd.DataFrame(results)
        try:
            saved = safe_save_df(df_tmp, out_path)
            print("  Saved interim results to", saved)
        except Exception as e:
            print("  Failed to save interim results:", e)

    # final save (with timestamp)
    df = pd.DataFrame(results)
    try:
        saved = safe_save_df(df, out_path)
        print("\nAll done. Final results saved to", saved)
    except Exception as e:
        print("ERROR saving final results:", e)

if __name__ == '__main__':
    main()