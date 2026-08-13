#!/usr/bin/env python3
"""
scripts/find_min_f_for_p90_auto.py

Automated 3-stage pipeline to find minimal volume fraction f such that conduction
probability p(f) >= target (default target 0.9).

Stages:
  1) Coarse scan (low-cost Monte Carlo) to find candidate transition interval.
  2) Refine with adaptive binary search (increase trials when inconclusive).
  3) High-precision confirmation at final candidate f.

Save intermediate CSVs for resume and auditing.

Usage example:
  python scripts/find_min_f_for_p90_auto.py --target 0.9 --fmin 0.001 --fmax 0.05 \
      --n_coarse 12 --coarse_trials 100 --initial_trials 200 --refine_max_trials 2000 \
      --final_trials 5000 --processes 4

Run from repository root (so src/ modules import correctly). On Windows run from terminal.
"""

import os, sys, time, math, argparse
from datetime import datetime
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

# Ensure repo root is in path (this file expected under repo/scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import project modules
try:
    from src.geometry import Cylinder
    from src.truncation import split_cylinder_by_box
    from src.connectivity import build_connectivity
except Exception as e:
    print("ERROR importing src modules:", e)
    raise

# ---------------- Utility functions ----------------
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

# Worker for a single trial (module-level for pickling)
def _run_single_trial(args):
    N, Lbox, rad, length, thresh, seed = args
    cyls = sample_random_cylinders(N, Lbox, rad, length, seed=seed)
    segments = []
    for c in cyls:
        segs = split_cylinder_by_box(c, Lbox)
        segments.extend(segs)
    connected, _uf = build_connectivity(segments, Lbox, thresh)
    return 1 if connected else 0

def estimate_p(f, trials, L, r, h, thresh, seed_base=0, processes=1, show_progress=False):
    N = compute_N_from_f(f, L, r, h)
    if trials <= 0 or N == 0:
        return {'f': f, 'N': N, 'trials': trials, 'successes': 0, 'p_hat': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'elapsed_s': 0.0}
    args_list = []
    for t in range(trials):
        seed = int(seed_base + t + (int(abs(f) * 1e9) % 1000000))
        args_list.append((N, L, r, h, thresh, seed))
    t0 = time.time()
    if processes == 1:
        if show_progress and _HAS_TQDM:
            results = [ _run_single_trial(a) for a in tqdm(args_list, desc=f"f={f:.6f}") ]
        else:
            results = [ _run_single_trial(a) for a in args_list ]
    else:
        with Pool(processes) as pool:
            if show_progress and _HAS_TQDM:
                results = []
                for res in tqdm(pool.imap(_run_single_trial, args_list), total=len(args_list), desc=f"f={f:.6f}"):
                    results.append(res)
            else:
                results = pool.map(_run_single_trial, args_list)
    successes = int(sum(results))
    p_hat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)
    elapsed = time.time() - t0
    return {'f': f, 'N': N, 'trials': trials, 'successes': successes, 'p_hat': p_hat, 'ci_low': ci_low, 'ci_high': ci_high, 'elapsed_s': elapsed}

# Safe save
def safe_save_df(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        df.to_csv(out_path, index=False)
        return out_path
    except PermissionError:
        alt = out_path.replace('.csv', f'_{int(time.time())}.csv')
        df.to_csv(alt, index=False)
        return alt

# ---------------- Automated pipeline ----------------
def coarse_scan(fmin, fmax, n_coarse, coarse_trials, L, r, h, thresh, seed_base, processes, out_coarse_csv, resume=True):
    """Run coarse scan; if resume and out_coarse_csv exists, load it and run missing f values."""
    if resume and os.path.exists(out_coarse_csv):
        try:
            df_exist = pd.read_csv(out_coarse_csv)
            done_fs = set(df_exist['f'].values.tolist())
            print(f"Loaded existing coarse results ({len(df_exist)} rows). Will skip existing f points.")
        except Exception:
            df_exist = None
            done_fs = set()
    else:
        df_exist = None
        done_fs = set()

    f_list = list(np.linspace(fmin, fmax, n_coarse))
    results = [] if df_exist is None else df_exist.to_dict('records')

    for f in f_list:
        if f in done_fs:
            print(f"skip f={f:.6f} (already present)")
            continue
        print(f"coarse: evaluating f={f:.6f} with trials={coarse_trials}")
        res = estimate_p(f, coarse_trials, L, r, h, thresh, seed_base=seed_base, processes=processes, show_progress=False)
        results.append(res)
        df_tmp = pd.DataFrame(results)
        safe_save_df(df_tmp, out_coarse_csv)
    return pd.DataFrame(results)

def locate_transition_interval(df_coarse, target, margin_frac=1.0, fallback_width=0.01):
    """From coarse results (DataFrame with f,p_hat), find a bracket [f_lo,f_hi] likely containing minimal f with p>=target.
    Strategy:
      - If any f has p_hat >= target: take first such f as f_hi, and f_lo previous f (or f/2 if none).
      - Else if p_hat crosses 0.5: take crossing interval.
      - Else take interval around largest slope (max diff).
    margin_frac: expand bracket by this fraction on each side.
    """
    df = df_coarse.sort_values('f').reset_index(drop=True)
    fs = df['f'].values
    p = df['p_hat'].values

    # if any >= target
    idxs = np.where(p >= target)[0]
    if len(idxs) > 0:
        hi_idx = idxs[0]
        hi = fs[hi_idx]
        lo = fs[hi_idx-1] if hi_idx > 0 else max(fs[0]/2.0, 1e-8)
        return max(1e-8, lo), min(1.0, hi)

    # try median crossing 0.5
    mid_thr = 0.5
    idxs_mid = np.where(p >= mid_thr)[0]
    if len(idxs_mid) > 0:
        hi_idx = idxs_mid[0]
        hi = fs[hi_idx]
        lo = fs[hi_idx-1] if hi_idx > 0 else max(fs[0]/2.0, 1e-8)
        return max(1e-8, lo), min(1.0, hi)

    # else pick largest slope
    diffs = np.diff(p)
    if len(diffs) == 0:
        # fallback small interval near max p
        fi = fs[np.argmax(p)]
        return max(1e-8, fi - fallback_width/2.0), min(1.0, fi + fallback_width/2.0)
    max_idx = np.argmax(diffs)
    lo = fs[max_idx]
    hi = fs[max_idx+1]
    # expand a bit
    width = hi - lo
    lo = max(1e-8, lo - margin_frac * width)
    hi = min(1.0, hi + margin_frac * width)
    return lo, hi

def adaptive_binary_search(target, f_lo, f_hi, L, r, h, thresh,
                           initial_trials, max_trials, processes, tol, seed_base, out_refine_csv):
    """Binary search minimal f with adaptive trials."""
    left = f_lo
    right = f_hi
    history = []

    while right - left > tol:
        mid = 0.5 * (left + right)
        trials = initial_trials
        print(f"Refine: testing mid={mid:.6g} trials={trials}")
        res = estimate_p(mid, trials, L, r, h, thresh, seed_base=seed_base, processes=processes)
        history.append(res)
        # decisive low CI >= target
        if res['ci_low'] >= target:
            right = mid
            print(f"  mid accepted by ci_low >= target ({res['ci_low']:.4f} >= {target})")
            safe_save_df(pd.DataFrame(history), out_refine_csv)
            continue
        # decisive high CI < target -> mid too small
        if res['ci_high'] < target:
            left = mid
            print(f"  mid rejected by ci_high < target ({res['ci_high']:.4f} < {target})")
            safe_save_df(pd.DataFrame(history), out_refine_csv)
            continue
        # inconclusive -> increase trials adaptively
        while trials < max_trials:
            trials = min(trials * 2, max_trials)
            print(f"  inconclusive, increasing trials -> {trials}")
            res = estimate_p(mid, trials, L, r, h, thresh, seed_base=seed_base, processes=processes)
            history.append(res)
            if res['ci_low'] >= target:
                right = mid
                print(f"  now accepted (ci_low={res['ci_low']:.4f})")
                break
            if res['ci_high'] < target:
                left = mid
                print(f"  now rejected (ci_high={res['ci_high']:.4f})")
                break
            # else continue until trials exhausted
        else:
            # reached max_trials and still inconclusive
            print(f"  reached max_trials={max_trials} and still inconclusive; use p_hat to decide")
            if res['p_hat'] >= target:
                right = mid
            else:
                left = mid
        safe_save_df(pd.DataFrame(history), out_refine_csv)

    # final candidate f = right (smallest f considered acceptable)
    candidate = right
    return candidate, history, (left, right)

def high_precision_check(f_candidate, final_trials, L, r, h, thresh, processes, seed_base):
    print(f"High-precision check at f={f_candidate} with trials={final_trials}")
    res = estimate_p(f_candidate, final_trials, L, r, h, thresh, seed_base=seed_base, processes=processes, show_progress=False)
    return res

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Automated pipeline to find minimal f for p(f) >= target")
    p.add_argument('--target', type=float, default=0.9, help='target conduction probability (default 0.9)')
    p.add_argument('--fmin', type=float, default=0.001, help='coarse f min')
    p.add_argument('--fmax', type=float, default=0.05, help='coarse f max')
    p.add_argument('--n_coarse', type=int, default=12, help='number of coarse points')
    p.add_argument('--coarse_trials', type=int, default=100, help='trials per coarse point')
    p.add_argument('--initial_trials', type=int, default=200, help='initial trials for refine midpoints')
    p.add_argument('--refine_max_trials', type=int, default=2000, help='max trials allowed when refining')
    p.add_argument('--final_trials', type=int, default=5000, help='high-precision trials at final candidate')
    p.add_argument('--processes', type=int, default=4, help='worker processes for Monte Carlo')
    p.add_argument('--thresh', type=float, default=1.8, help='surface-to-surface threshold (nm)')
    p.add_argument('--L', type=float, default=10000.0, help='box size nm')
    p.add_argument('--r', type=float, default=30.0, help='cylinder radius nm')
    p.add_argument('--h', type=float, default=5000.0, help='cylinder length nm')
    p.add_argument('--tol', type=float, default=5e-4, help='tolerance for f convergence')
    p.add_argument('--seed', type=int, default=20260812, help='seed base')
    p.add_argument('--out_prefix', type=str, default=os.path.join('results','outputs','find_min_f_auto'), help='output file prefix (CSV files will be created)')
    p.add_argument('--resume', action='store_true', help='resume coarse scan if coarse CSV exists')
    return p.parse_args()

def main():
    args = parse_args()
    print("Automated pipeline start:", datetime.utcnow().isoformat())
    print("Parameters:", args)

    # multiproc method for Windows
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    out_coarse_csv = args.out_prefix + "_coarse.csv"
    out_refine_csv = args.out_prefix + "_refine_history.csv"
    out_final_csv  = args.out_prefix + "_final.csv"

    # 1) coarse scan
    df_coarse = coarse_scan(args.fmin, args.fmax, args.n_coarse, args.coarse_trials,
                            args.L, args.r, args.h, args.thresh, args.seed, args.processes, out_coarse_csv, resume=args.resume)
    print("Coarse scan done. Summary:")
    print(df_coarse[['f','N','p_hat','ci_low','ci_high']])

    # 2) locate transition interval
    f_lo, f_hi = locate_transition_interval(df_coarse, args.target)
    print(f"Located transition interval: [{f_lo:.6g}, {f_hi:.6g}]")

    # 3) refine (adaptive binary search)
    candidate_f, history, bracket = adaptive_binary_search(args.target, f_lo, f_hi,
                                                           args.L, args.r, args.h, args.thresh,
                                                           args.initial_trials, args.refine_max_trials,
                                                           args.processes, args.tol, args.seed, out_refine_csv)
    print(f"Candidate f (refine result): {candidate_f:.6g}")

    # 4) high-precision confirmation
    final_res = high_precision_check(candidate_f, args.final_trials, args.L, args.r, args.h, args.thresh, args.processes, args.seed)
    print("Final high-precision result:", final_res)

    # Save final
    df_final = pd.DataFrame([final_res])
    saved = safe_save_df(df_final, out_final_csv)
    print("Saved final result to", saved)

    # Summary file
    summary = {
        'target': args.target,
        'candidate_f': candidate_f,
        'candidate_trials_used': history[-1]['trials'] if history else None,
        'final_p_hat': final_res['p_hat'],
        'final_ci_low': final_res['ci_low'],
        'final_ci_high': final_res['ci_high'],
        'timestamp': datetime.utcnow().isoformat()
    }
    sum_path = args.out_prefix + "_summary.csv"
    safe_save_df(pd.DataFrame([summary]), sum_path)
    print("Saved summary to", sum_path)
    print("Pipeline complete at", datetime.utcnow().isoformat())

if __name__ == '__main__':
    main()