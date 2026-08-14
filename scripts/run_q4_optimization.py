#!/usr/bin/env python3
"""
scripts/run_q4_optimization.py

2D grid search (coarse -> local refine) for minimum cost under constraint p(f_A,f_B) >= threshold.
Uses src/ modules and real Monte Carlo (no proxy).

Supports multiprocessing and resume from CSV.

Example:
  python scripts/run_q4_optimization.py --fA_min 0.0 --fA_max 0.005 --fB_min 0.0 --fB_max 0.01 \
    --coarse_na 8 --coarse_nb 8 --coarse_trials 500 --fine_trials 2000 --out_dir results/outputs --processes 12
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
from typing import Optional

# add repo root to path
repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.geometry import Cylinder, Sphere
from src.monte_carlo import run_single_trial_mixed
from src.truncation import split_cylinder_by_box, split_sphere_by_box
from src.connectivity import build_connectivity

# Physical params (problem statement)
L = 10000.0  # nm
r_A = 30.0   # nm (cylinder radius)
h_A = 5000.0 # nm (cylinder length)
r_B = 200.0  # nm (sphere radius)
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

# Worker for multiprocessing map; must be at top-level for pickling
def _trial_worker(args):
    """args: tuple(N_A, N_B, L, r_A, h_A, r_B, thresh, seed) -> int (1 for connected else 0)"""
    N_A, N_B, Lloc, rAloc, hAloc, rBloc, threshloc, seed = args
    connected, _, _ = run_single_trial_mixed(N_A, N_B, Lloc, rAloc, hAloc, rBloc, threshloc, seed=seed)
    return 1 if connected else 0

def evaluate_point(fA, fB, trials, seed_base, processes:int=4):
    """Evaluate a single (fA,fB) by running 'trials' monte-carlo trials, possibly in parallel."""
    N_A = f_to_N(fA, V_A_single_nm3)
    N_B = f_to_N(fB, V_B_single_nm3)
    successes = 0
    t0 = time.time()
    if trials <= 0:
        p_hat = 0.0
        ci_low = 0.0
        ci_high = 0.0
        elapsed = 0.0
    else:
        args_list = []
        for t in range(trials):
            # construct deterministic-ish seed from seed_base, index, and f values
            seed = int(seed_base + t + (int(abs(fA) * 1e9) % 1000000) + (int(abs(fB) * 1e6) % 100000))
            args_list.append((N_A, N_B, L, r_A, h_A, r_B, thresh, seed))
        if processes is None or processes <= 1:
            # serial
            results = [_trial_worker(a) for a in args_list]
        else:
            # cap processes to CPU count
            max_procs = max(1, min(processes, multiprocessing.cpu_count()))
            with multiprocessing.Pool(processes=max_procs) as pool:
                results = pool.map(_trial_worker, args_list)
        successes = int(sum(results))
        p_hat = successes / trials
        ci_low, ci_high = wilson_interval(successes, trials)
        elapsed = time.time() - t0
    cost = compute_cost(fA, fB)
    return {'fA': fA, 'fB': fB, 'N_A': N_A, 'N_B': N_B, 'trials': trials, 'successes': successes,
            'p_hat': p_hat, 'ci_low': ci_low, 'ci_high': ci_high, 'cost': cost, 'elapsed_s': elapsed}


def coarse_grid_search(fA_min, fA_max, fB_min, fB_max, nA, nB, trials, seed_base, out_csv, processes=4, resume_df:Optional[pd.DataFrame]=None):
    fA_list = np.linspace(fA_min, fA_max, nA)
    fB_list = np.linspace(fB_min, fB_max, nB)
    records = []
    # if resume_df provided, pre-load existing records into 'records' to preserve order
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
            res = evaluate_point(fa, fb, trials, seed_base, processes=processes)
            records.append(res)
            pd.DataFrame(records).to_csv(out_csv, index=False)
            print("  p_hat={:.4f} ci=({:.4f},{:.4f}) cost={:.4f}元 elapsed={:.1f}s".format(res['p_hat'], res['ci_low'], res['ci_high'], res['cost'], res['elapsed_s']))
    return pd.DataFrame(records)


def refine_near_candidates(df_coarse, threshold, fine_trials, seed_base, out_csv, processes=4, resume_df:Optional[pd.DataFrame]=None):
    # select candidate points where p_hat >= threshold or ci_low near threshold
    candidates = df_coarse[(df_coarse['p_hat'] >= threshold) | (df_coarse['ci_low'] >= threshold - 0.02)]
    if candidates.empty:
        # fallback: pick top few by p_hat/cost ratio
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
                res = evaluate_point(fa, fb, fine_trials, seed_base, processes=processes)
                fine_records.append(res)
                pd.DataFrame(fine_records).to_csv(out_csv, index=False)
                print("  p_hat={:.4f} ci=({:.4f},{:.4f}) cost={:.4f}元".format(res['p_hat'], res['ci_low'], res['ci_high'], res['cost']))
    return pd.DataFrame(fine_records)


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

    # Stage: coarse
    df_coarse = coarse_grid_search(args.fA_min, args.fA_max, args.fB_min, args.fB_max,
                                   args.coarse_na, args.coarse_nb, args.coarse_trials,
                                   args.seed, coarse_csv, processes=args.processes, resume_df=resume_df)

    # Stage: fine
    df_fine = refine_near_candidates(df_coarse, args.threshold, args.fine_trials, args.seed,
                                     fine_csv, processes=args.processes, resume_df=resume_df)

    # combine and find feasible min-cost
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

if __name__ == "__main__":
    # set start method for multiprocessing to spawn (safer across platforms)
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass
    main()
