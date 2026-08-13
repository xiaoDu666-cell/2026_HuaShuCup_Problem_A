#!/usr/bin/env python3
"""
scripts/run_p90_search.py

Two-stage search for minimal volume fraction f such that conduction probability p(f) >= threshold.

Stage 1 (coarse):
  - sweep f in [coarse_f_min, coarse_f_max] with coarse_npoints points
  - run coarse_trials per f
  - save CSV and p(f) plot

Stage 2 (fine):
  - bracket transition interval from coarse results
  - scan from f_lo to f_hi with step fine_step_percent (in percent units -> converted to fraction)
    e.g., fine_step_percent = 0.001 means 0.001% -> 0.00001 fraction step
  - run fine_trials per f until first f meets requirement (p_hat >= threshold or ci_low >= threshold if --require-ci-low)
  - save fine CSV and final answer CSV

Usage (example):
  python scripts/run_p90_search.py \
      --coarse_f_min 0.0001 --coarse_f_max 0.01 --coarse_npoints 40 \
      --coarse_trials 400 --fine_trials 2000 --fine_step_percent 0.001 \
      --processes 8

Notes:
 - Run from repo root so src/ modules are importable.
 - In Jupyter set --processes 1 to avoid multiprocessing issues.
"""
import os
import sys
import math
import time
import argparse
from pathlib import Path
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd

# Optional progress bar
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# -----------------------
# Default parameters (change here or use CLI)
# -----------------------
# Geometry / physics
L = 10000.0         # box edge length (nm)
r = 30.0            # cylinder radius (nm)
h = 5000.0          # cylinder length (nm)
thresh = 1.8        # surface-to-surface threshold (nm)

# Coarse sweep parameters
coarse_f_min = 0.0001      # fraction (e.g., 0.0001 = 0.01%)
coarse_f_max = 0.02        # fraction (e.g., 0.02 = 2%)
coarse_npoints = 50
coarse_trials = 400        # trials per coarse point

# Fine scan parameters
fine_step_percent = 0.001  # step in percent (0.001%); converted to fraction: /100
fine_trials = 2000         # trials per fine point (2000-5000 recommended)

# Threshold and stats
threshold = 0.90           # target probability
require_ci_low = False     # if True require Wilson ci_low >= threshold; else require p_hat >= threshold

# Parallel / seeds / outputs
processes = 4
seed_base = 20260812

out_dir = Path("results") / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
coarse_csv = out_dir / "coarse_scan_p90.csv"
fine_csv = out_dir / "fine_scan_p90.csv"
final_csv = out_dir / "final_p90_answer.csv"
coarse_png = out_dir / "coarse_scan_p90.png"

# -----------------------
# Utility functions and wrappers (reuse src/ modules)
# -----------------------
# Ensure repo root is on path
repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import project modules (must exist)
try:
    from src.geometry import Cylinder
    from src.truncation import split_cylinder_by_box
    from src.connectivity import build_connectivity
except Exception as e:
    print("ERROR importing project modules from src/:", e)
    raise

def volume_cylinder(rad, length):
    return math.pi * (rad ** 2) * length

def compute_N_from_f(f, Lbox, rad, length):
    V_box = Lbox ** 3
    V_cyl = volume_cylinder(rad, length)
    if V_cyl <= 0:
        raise ValueError("Cylinder volume must be positive")
    return max(0, int(round((f * V_box) / V_cyl)))

def sample_random_cylinders(N, Lbox, rad, length, seed=None):
    """Randomly sample N cylinders (center uniform in box, direction uniform on sphere)."""
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

# Wilson interval
def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + (z**2) / n
    centre = phat + (z**2) / (2 * n)
    adj = z * math.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

# Worker for single trial
def _run_single_trial(args):
    N, Lbox, rad, length, thr, seed = args
    cyls = sample_random_cylinders(N, Lbox, rad, length, seed=seed)
    segments = []
    for c in cyls:
        segments.extend(split_cylinder_by_box(c, Lbox))
    connected, _uf = build_connectivity(segments, Lbox, thr)
    return 1 if connected else 0

def estimate_p_for_f(f, trials, Lbox, rad, length, thr, seed_base, processes=1, show_progress=False):
    """Estimate p(f) by Monte Carlo with given trials and processes; return dict with stats."""
    N = compute_N_from_f(f, Lbox, rad, length)
    if trials <= 0 or N == 0:
        return {'f': f, 'N': N, 'trials': trials, 'successes': 0, 'p_hat': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'elapsed_s': 0.0}
    args_list = []
    for t in range(trials):
        seed = int(seed_base + t + (int(abs(f) * 1e9) % 1000000))
        args_list.append((N, Lbox, rad, length, thr, seed))
    t0 = time.time()
    if processes == 1:
        if show_progress and _HAS_TQDM:
            results = [ _run_single_trial(a) for a in tqdm(args_list, desc=f"f={f:.6g}") ]
        else:
            results = [ _run_single_trial(a) for a in args_list ]
    else:
        with Pool(processes) as pool:
            if show_progress and _HAS_TQDM:
                results = []
                for r in tqdm(pool.imap(_run_single_trial, args_list), total=len(args_list), desc=f"f={f:.6g}"):
                    results.append(r)
            else:
                results = pool.map(_run_single_trial, args_list)
    successes = int(sum(results))
    p_hat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)
    return {'f': f, 'N': N, 'trials': trials, 'successes': successes, 'p_hat': p_hat, 'ci_low': ci_low, 'ci_high': ci_high, 'elapsed_s': time.time() - t0}

# -----------------------
# Coarse scan
# -----------------------
def run_coarse_scan(f_min, f_max, npoints, trials, Lbox, rad, length, thr, seed_base, processes, out_csv, out_png):
    f_list = np.linspace(f_min, f_max, npoints)
    records = []
    print(f"Coarse scan: {npoints} points from {f_min:.6g} to {f_max:.6g}, {trials} trials each.")
    for f in f_list:
        print(f"Coarse: evaluating f={f:.6g} ...")
        res = estimate_p_for_f(f, trials, Lbox, rad, length, thr, seed_base, processes=processes, show_progress=False)
        records.append(res)
        # save incremental
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"  f={f:.6g}, N={res['N']}, successes={res['successes']}/{res['trials']}, p_hat={res['p_hat']:.4f}, ci=({res['ci_low']:.4f},{res['ci_high']:.4f}), elapsed={res['elapsed_s']:.1f}s")
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    # plot
    try:
        import matplotlib.pyplot as plt
        x_pct = df['f'].values * 100.0
        y = df['p_hat'].values
        low = df['ci_low'].values
        high = df['ci_high'].values
        yerr_lower = y - low
        yerr_upper = high - y
        fig, ax = plt.subplots(figsize=(8,4))
        ax.errorbar(x_pct, y, yerr=[yerr_lower, yerr_upper], fmt='o-', capsize=3)
        ax.axhline(threshold, color='red', linestyle='--', label=f"target {threshold:.2f}")
        ax.set_xlabel("体积分数 f (%)")
        ax.set_ylabel("导通概率 p(f)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        print("Saved coarse plot to", out_png)
    except Exception as e:
        print("Plotting coarse scan failed:", e)
    return df

# -----------------------
# Locate bracket from coarse results
# -----------------------
def locate_bracket_from_coarse(df_coarse, threshold, require_ci_low=False):
    df = df_coarse.sort_values('f').reset_index(drop=True)
    fs = df['f'].values
    phats = df['p_hat'].values
    ci_lows = df['ci_low'].values if 'ci_low' in df.columns else phats - 0.02
    # find first index where condition met
    for i, f in enumerate(fs):
        cond = (ci_lows[i] >= threshold) if require_ci_low else (phats[i] >= threshold)
        if cond:
            hi = f
            lo = fs[i-1] if i > 0 else max(f/2.0, 1e-12)
            return max(1e-12, lo), hi
    # not found
    return None, None

# -----------------------
# Fine scan inside bracket
# -----------------------
def run_fine_scan(f_lo, f_hi, fine_step_percent, trials, Lbox, rad, length, thr, seed_base, processes, out_csv, threshold=0.9, require_ci_low=False):
    # convert step percent to fraction: e.g., 0.001% -> 0.00001 fraction
    step_frac = (fine_step_percent / 100.0)
    # ensure we include f_hi
    f = f_lo
    records = []
    print(f"Fine scan from {f_lo:.6g} to {f_hi:.6g} step {step_frac:.6g} (fraction), trials={trials}")
    # Guard against infinite loops due to too small step
    max_iters = int(math.ceil((f_hi - f_lo) / step_frac)) + 5
    it = 0
    found_f = None
    while f <= f_hi + 1e-15 and it < max_iters:
        it += 1
        f_rounded = float(np.round(f, 12))  # avoid tiny floating drift
        print(f"Fine: evaluating f={f_rounded:.8g} ...")
        res = estimate_p_for_f(f_rounded, trials, Lbox, rad, length, thr, seed_base, processes=processes, show_progress=False)
        records.append(res)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"  f={f_rounded:.8g}, N={res['N']}, successes={res['successes']}/{res['trials']}, p_hat={res['p_hat']:.4f}, ci=({res['ci_low']:.4f},{res['ci_high']:.4f}), elapsed={res['elapsed_s']:.1f}s")
        cond = (res['ci_low'] >= threshold) if require_ci_low else (res['p_hat'] >= threshold)
        if cond:
            found_f = res
            print(f"--> Found first f meeting requirement at f={f_rounded:.8g}")
            break
        f += step_frac
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return found_f, df

# -----------------------
# Main routine and argument parsing
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Two-stage search for minimal f with p(f) >= threshold")
    p.add_argument("--coarse_f_min", type=float, default=coarse_f_min)
    p.add_argument("--coarse_f_max", type=float, default=coarse_f_max)
    p.add_argument("--coarse_npoints", type=int, default=coarse_npoints)
    p.add_argument("--coarse_trials", type=int, default=coarse_trials)
    p.add_argument("--fine_step_percent", type=float, default=fine_step_percent, help="step in percent (e.g., 0.001 means 0.001%%)")
    p.add_argument("--fine_trials", type=int, default=fine_trials)
    p.add_argument("--threshold", type=float, default=threshold)
    p.add_argument("--require_ci_low", action="store_true", help="require Wilson ci_low >= threshold for acceptance")
    p.add_argument("--processes", type=int, default=processes)
    p.add_argument("--seed", type=int, default=seed_base)
    p.add_argument("--out_dir", type=str, default=str(out_dir))
    p.add_argument("--coarse_csv", type=str, default=str(coarse_csv))
    p.add_argument("--fine_csv", type=str, default=str(fine_csv))
    p.add_argument("--final_csv", type=str, default=str(final_csv))
    p.add_argument("--coarse_png", type=str, default=str(coarse_png))
    return p.parse_args()

def main():
    args = parse_args()
    # override globals with args where needed
    fmin = args.coarse_f_min
    fmax = args.coarse_f_max
    npoints = args.coarse_npoints
    c_trials = args.coarse_trials
    step_percent = args.fine_step_percent
    f_trials = args.fine_trials
    thr_prob = args.threshold
    req_ci = args.require_ci_low
    procs = args.processes
    seed = args.seed
    out_dir_local = Path(args.out_dir)
    out_dir_local.mkdir(parents=True, exist_ok=True)
    coarse_csv_local = Path(args.coarse_csv)
    fine_csv_local = Path(args.fine_csv)
    final_csv_local = Path(args.final_csv)
    coarse_png_local = Path(args.coarse_png)

    # set start method for multiprocessing on platforms that need it
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    print("Run parameters:")
    print(" L, r, h, thresh:", L, r, h, thresh)
    print(" coarse f range:", fmin, "->", fmax, " npoints=", npoints, "trials=", c_trials)
    print(" fine step (percent):", step_percent, " fine_trials=", f_trials)
    print(" threshold:", thr_prob, " require_ci_low:", req_ci, " processes:", procs)
    print(" outputs:", coarse_csv_local, fine_csv_local, final_csv_local)

    # Stage 1: coarse scan
    df_coarse = run_coarse_scan(fmin, fmax, npoints, c_trials, L, r, h, thresh, seed, procs, coarse_csv_local, coarse_png_local)

    # Locate bracket
    f_lo, f_hi = locate_bracket_from_coarse(df_coarse, thr_prob, require_ci_low=req_ci)
    if f_lo is None:
        print("Warning: did not locate bracket where coarse p(f) reaches threshold. Consider expanding coarse range or increasing coarse_trials.")
        # fallback: take top interval near maximum p
        # pick point with maximum slope or maximum p
        idx_max = int(np.argmax(df_coarse['p_hat'].values))
        f_hi = float(df_coarse['f'].values[idx_max])
        f_lo = float(max(df_coarse['f'].values[0], f_hi - 0.001))
        print("Fallback bracket:", f_lo, f_hi)

    print("Using bracket for fine scan: [{:.8g}, {:.8g}]".format(f_lo, f_hi))

    # Stage 2: fine scan
    found_res, df_fine = run_fine_scan(f_lo, f_hi, step_percent, f_trials, L, r, h, thresh, seed, procs, fine_csv_local, threshold=thr_prob, require_ci_low=req_ci)

    # Save final answer
    if found_res is not None:
        final_record = {
            'found': True,
            'f': found_res['f'],
            'N': found_res['N'],
            'trials': found_res['trials'],
            'successes': found_res['successes'],
            'p_hat': found_res['p_hat'],
            'ci_low': found_res['ci_low'],
            'ci_high': found_res['ci_high'],
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        pd.DataFrame([final_record]).to_csv(final_csv_local, index=False)
        print("Final answer saved to", final_csv_local)
        print("Final answer:", final_record)
    else:
        final_record = {
            'found': False,
            'note': 'No f in fine grid reached threshold',
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        pd.DataFrame([final_record]).to_csv(final_csv_local, index=False)
        print("No f reached threshold in fine scan. Details saved to", final_csv_local)

if __name__ == "__main__":
    main()