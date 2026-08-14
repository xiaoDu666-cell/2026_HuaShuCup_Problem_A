#!/usr/bin/env python3
"""
scripts/run_q4_optimization.py

2D grid search (coarse -> local refine) for minimum cost under constraint p(f_A,f_B) >= threshold.
Uses src/ modules and real Monte Carlo (no proxy).

Optimizations:
 - Reuse a single multiprocessing.Pool across the whole scan (created in main)
 - Batch trials per pool task to reduce Pool.map overhead and RNG creation
 - Each batch-worker reuses a single RNG for its trials

Example:
  python scripts/run_q4_optimization.py --coarse_na 8 --coarse_nb 8 --coarse_trials 500 \
    --fine_trials 2000 --processes 12 --batch_size 16
"""
import os
import sys
import math
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing
from typing import Optional, Tuple

# add repo root to path
repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import specific utilities to call them directly inside worker (avoid extra indirection)
from src.geometry import Cylinder, Sphere
from src.monte_carlo import sample_random_cylinders, sample_random_spheres
from src.truncation import split_cylinder_by_box, split_sphere_by_box
from src.connectivity import build_connectivity

# Problem params
L = 10000.0  # nm
r_A = 30.0   # nm
h_A = 5000.0 # nm
r_B = 200.0  # nm
thresh = 1.8 # nm

# costs (元/μm^3)
cA = 1.05
cB = 0.05

def volume_cylinder_nm3(rad, length):
    return math.pi * (rad ** 2) * length

def volume_sphere_nm3(rad):
    return (4.0/3.0) * math.pi * (rad ** 3)

V_box_nm3 = L**3
V_A_single_nm3 = volume_cylinder_nm3(r_A, h_A)
V_B_single_nm3 = volume_sphere_nm3(r_B)

def f_to_N(f, V_single_nm3):
    return int(round((f * V_box_nm3) / V_single_nm3))

def compute_cost(fA, fB):
    V_A_total_um3 = (fA * V_box_nm3) / 1e9
    V_B_total_um3 = (fB * V_box_nm3) / 1e9
    return cA * V_A_total_um3 + cB * V_B_total_um3

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

# --------------------
# Batch worker
# --------------------
def _trial_batch_worker(args: Tuple[int,int,float,float,float,float,float,float,int,int]) -> int:
    """
    Execute n_trials Monte-Carlo trials in one process and return number of successes.

    args is tuple:
      (N_A, N_B, L, r_A, h_A, r_B, thresh, seed_start, n_trials)
    """
    N_A, N_B, Lloc, rAloc, hAloc, rBloc, threshloc, seed_start, n_trials = args
    rng = np.random.default_rng(int(seed_start))
    successes = 0
    # local references for speed
    _sample_cyl = sample_random_cylinders
    _sample_sph = sample_random_spheres
    _split_cyl = split_cylinder_by_box
    _split_sph = split_sphere_by_box
    _build_conn = build_connectivity

    for _ in range(n_trials):
        # sample particles using the same rng (will advance RNG state)
        cyls = _sample_cyl(N_A, Lloc, rAloc, hAloc, start_id=0, rng=rng)
        sphs = _sample_sph(N_B, Lloc, rBloc, start_id=N_A, rng=rng)
        particles = []
        for c in cyls:
            particles.extend(_split_cyl(c, Lloc))
        for s in sphs:
            particles.extend(_split_sph(s, Lloc))
        connected, _ = _build_conn(particles, Lloc, threshloc)
        if connected:
            successes += 1
    return int(successes)

# --------------------
# Evaluate point with batching (uses provided pool if non-None)
# --------------------
def evaluate_point(fA, fB, trials, seed_base, processes:int=4, batch_size:int=0, pool: Optional[multiprocessing.pool.Pool]=None):
    """
    Evaluate (fA, fB) by splitting trials into batches and mapping them to pool (if provided).
    batch_size: trials per batch/task. If 0 or None, default set to max(1, processes * 4).
    """
    N_A = f_to_N(fA, V_A_single_nm3)
    N_B = f_to_N(fB, V_B_single_nm3)
    t0 = time.time()

    if batch_size is None or batch_size <= 0:
        batch_size = max(1, (processes or 1) * 4)

    if trials <= 0:
        p_hat = 0.0; ci_low = 0.0; ci_high = 0.0; elapsed = 0.0
        return {'fA': fA, 'fB': fB, 'N_A': N_A, 'N_B': N_B, 'trials': trials, 'successes': 0,
                'p_hat': p_hat, 'ci_low': ci_low, 'ci_high': ci_high, 'cost': compute_cost(fA,fB), 'elapsed_s': elapsed}

    # prepare batch args
    n_batches = int(math.ceil(trials / batch_size))
    batch_args = []
    remaining = trials
    # use different seed_start for each batch for reproducibility
    for b in range(n_batches):
        nb = min(batch_size, remaining)
        seed_start = int(seed_base + b * 1000003 + (int(abs(fA) * 1e9) % 1000000) + (int(abs(fB) * 1e6) % 100000))
        batch_args.append((N_A, N_B, L, r_A, h_A, r_B, thresh, seed_start, nb))
        remaining -= nb

    # run batches
    successes = 0
    if pool is not None:
        # pool.map returns list of ints (successes per batch)
        results = pool.map(_trial_batch_worker, batch_args)
        successes = sum(results)
    else:
        # no pool: run serially
        for a in batch_args:
            successes += _trial_batch_worker(a)

    p_hat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)
    elapsed = time.time() - t0
    return {'fA': fA, 'fB': fB, 'N_A': N_A, 'N_B': N_B, 'trials': trials, 'successes': successes,
            'p_hat': p_hat, 'ci_low': ci_low, 'ci_high': ci_high, 'cost': compute_cost(fA, fB), 'elapsed_s': elapsed}

# --------------------
# Grid search & refine (accept pool and batch_size)
# --------------------
def coarse_grid_search(fA_min, fA_max, fB_min, fB_max, nA, nB, trials, seed_base, out_csv, processes=4, resume_df:Optional[pd.DataFrame]=None, pool: Optional[multiprocessing.pool.Pool]=None, batch_size:int=0):
    fA_list = np.linspace(fA_min, fA_max, nA)
    fB_list = np.linspace(fB_min, fB_max, nB)
    records = []
    existing = set()
    if resume_df is not None and not resume_df.empty:
        for _, row in resume_df.iterrows():
            try:
                fa = float(row['fA']); fb = float(row['fB'])
            except Exception:
                continue
            existing.add((round(fa,12), round(fb,12)))
            records.append(row.to_dict())

    for fa in fA_list:
        for fb in fB_list:
            key = (round(float(fa),12), round(float(fb),12))
            if key in existing:
                print(f"Skipping already computed fA={fa:.6g}, fB={fb:.6g}")
                continue
            print(f"Coarse evaluating fA={fa:.6g}, fB={fb:.6g} ...")
            res = evaluate_point(fa, fb, trials, seed_base, processes=processes, batch_size=batch_size, pool=pool)
            records.append(res)
            pd.DataFrame(records).to_csv(out_csv, index=False)
            print("  p_hat={:.4f} ci=({:.4f},{:.4f}) cost={:.4f}元 elapsed={:.1f}s".format(res['p_hat'], res['ci_low'], res['ci_high'], res['cost'], res['elapsed_s']))
    return pd.DataFrame(records)

def refine_near_candidates(df_coarse, threshold, fine_trials, seed_base, out_csv, processes=4, resume_df:Optional[pd.DataFrame]=None, pool: Optional[multiprocessing.pool.Pool]=None, batch_size:int=0):
    candidates = df_coarse[(df_coarse['p_hat'] >= threshold) | (df_coarse['ci_low'] >= threshold - 0.02)]
    if candidates.empty:
        df_coarse = df_coarse.copy()
        df_coarse['score'] = df_coarse['p_hat'] / (df_coarse['cost'] + 1e-12)
        candidates = df_coarse.sort_values('score', ascending=False).head(6)
    fine_records = []
    existing = set()
    if resume_df is not None and not resume_df.empty:
        for _, row in resume_df.iterrows():
            try:
                fa = float(row['fA']); fb = float(row['fB'])
            except Exception:
                continue
            existing.add((round(fa,12), round(fb,12)))
            fine_records.append(row.to_dict())

    for _, row in candidates.iterrows():
        fa0 = float(row['fA']); fb0 = float(row['fB'])
        da = max(0.0005, fa0 * 0.25 + 1e-6)
        db = max(0.001, fb0 * 0.25 + 1e-6)
        fa_list = np.linspace(max(0.0, fa0 - da), fa0 + da, 9)
        fb_list = np.linspace(max(0.0, fb0 - db), fb0 + db, 9)
        for fa in fa_list:
            for fb in fb_list:
                key = (round(float(fa),12), round(float(fb),12))
                if key in existing:
                    print(f"Skipping already computed fine fA={fa:.6g}, fB={fb:.6g}")
                    continue
                print(f"Fine evaluating fA={fa:.6g}, fB={fb:.6g} ...")
                res = evaluate_point(fa, fb, fine_trials, seed_base, processes=processes, batch_size=batch_size, pool=pool)
                fine_records.append(res)
                pd.DataFrame(fine_records).to_csv(out_csv, index=False)
                print("  p_hat={:.4f} ci=({:.4f},{:.4f}) cost={:.4f}元".format(res['p_hat'], res['ci_low'], res['ci_high'], res['cost']))
    return pd.DataFrame(fine_records)

# --------------------
# CLI and main
# --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fA_min", type=float, default=0.0)
    p.add_argument("--fA_max", type=float, default=0.005)
    p.add_argument("--fB_min", type=float, default=0.0)
    p.add_argument("--fB_max", type=float, default=0.01)
    p.add_argument("--coarse_na", type=int, default=8)
    p.add_argument("--coarse_nb", type=int, default=8)
    p.add_argument("--coarse_trials", type=int, default=500)
    p.add_argument("--fine_trials", type=int, default=2000)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--out_dir", type=str, default="results/outputs")
    p.add_argument("--processes", type=int, default=4, help="Number of worker processes for Monte Carlo trials (up to cpu_count)")
    p.add_argument("--batch_size", type=int, default=0, help="Trials per batch task; default processes*4 if 0")
    p.add_argument("--resume", type=str, default=None, help="Path to existing CSV to resume from (coarse or fine results CSV)")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    coarse_csv = out_dir / "q4_coarse_results.csv"
    fine_csv = out_dir / "q4_fine_results.csv"
    final_csv = out_dir / "q4_best_answer.csv"

    resume_df = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            try:
                resume_df = pd.read_csv(resume_path)
                print("Loaded resume CSV with {} rows".format(len(resume_df)))
            except Exception as e:
                print("Failed to load resume CSV:", e)
                resume_df = None
        else:
            print("Resume path not found, continuing without resume:", resume_path)

    pool = None
    if args.processes is not None and args.processes > 1:
        max_procs = max(1, min(args.processes, multiprocessing.cpu_count()))
        print(f"Creating multiprocessing pool with {max_procs} workers")
        pool = multiprocessing.Pool(processes=max_procs)
    else:
        print("Running without multiprocessing pool (serial execution)")

    try:
        df_coarse = coarse_grid_search(args.fA_min, args.fA_max, args.fB_min, args.fB_max,
                                       args.coarse_na, args.coarse_nb, args.coarse_trials,
                                       args.seed, coarse_csv, processes=args.processes, resume_df=resume_df, pool=pool, batch_size=args.batch_size)

        df_fine = refine_near_candidates(df_coarse, args.threshold, args.fine_trials, args.seed,
                                         fine_csv, processes=args.processes, resume_df=resume_df, pool=pool, batch_size=args.batch_size)

        df_all = pd.concat([df_coarse, df_fine], ignore_index=True)
        feasible = df_all[df_all['p_hat'] >= args.threshold]
        if not feasible.empty:
            best = feasible.loc[feasible['cost'].idxmin()]
            best_record = best.to_dict()
            best_record['found'] = True
        else:
            feasible2 = df_all[df_all['ci_low'] >= args.threshold]
            if not feasible2.empty:
                best = feasible2.loc[feasible2['cost'].idxmin()]
                best_record = best.to_dict()
                best_record['found'] = True
            else:
                best_record = {'found': False, 'note': 'No feasible point found; consider larger search or more trials.'}

        pd.DataFrame([best_record]).to_csv(final_csv, index=False)
        print("Best record saved to", final_csv)
        print(best_record)
    finally:
        if pool is not None:
            print("Closing multiprocessing pool")
            pool.close()
            pool.join()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass
    main()