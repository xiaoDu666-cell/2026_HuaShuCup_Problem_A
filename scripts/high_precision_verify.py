#!/usr/bin/env python3
"""
High-precision verification for single f.
Saves per-trial seeds/results and sample pickles for later visualization.

Usage:
  python scripts/high_precision_verify.py --f 0.001 --trials 10000 --processes 8 --save-success 5 --save-fail 5
"""
import os, sys, time, math, argparse, pickle
from multiprocessing import Pool
import multiprocessing
from datetime import datetime
import numpy as np
import pandas as pd

# ensure repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.geometry import Cylinder
from src.truncation import split_cylinder_by_box
from src.connectivity import build_connectivity

def volume_cylinder(rad, length):
    return math.pi * (rad ** 2) * length

def compute_N_from_f(f, Lbox, rad, length):
    V_box = Lbox ** 3
    V_cyl = volume_cylinder(rad, length)
    if V_cyl <= 0:
        raise ValueError("Cylinder volume must be positive")
    return max(0, int(round((f * V_box) / V_cyl)))

def sample_random_cylinders(N, Lbox, rad, length, seed=None):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-Lbox/2, Lbox/2, size=(N,3))
    dirs = rng.normal(size=(N,3))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    dirs = dirs / norms
    half = length/2.0
    cyls=[]
    for i in range(N):
        c = centers[i]; d = dirs[i]
        p0 = c - d*half; p1 = c + d*half
        cyls.append(Cylinder(p0, p1, rad, id=i))
    return cyls

def trial_worker(args):
    # args = (seed, N, L, r, h, thresh)
    seed, N, L, r, h, thresh = args
    t0 = time.time()
    cyls = sample_random_cylinders(N, L, r, h, seed=seed)
    segments=[]
    for c in cyls:
        segments.extend(split_cylinder_by_box(c, L))
    connected, _ = build_connectivity(segments, L, thresh)
    elapsed = time.time() - t0
    return {'seed': int(seed), 'connected': bool(connected), 'elapsed_s': elapsed, 'segments': segments if (connected or False) else None}

def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = k/n
    denom = 1 + (z**2)/n
    centre = phat + (z**2)/(2*n)
    adj = z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--f', type=float, required=True)
    p.add_argument('--trials', type=int, default=10000)
    p.add_argument('--processes', type=int, default=4)
    p.add_argument('--L', type=float, default=10000.0)
    p.add_argument('--r', type=float, default=30.0)
    p.add_argument('--h', type=float, default=5000.0)
    p.add_argument('--thresh', type=float, default=1.8)
    p.add_argument('--seed_base', type=int, default=20260812)
    p.add_argument('--save-success', type=int, default=5, help='save this many success examples (pickle)')
    p.add_argument('--save-fail', type=int, default=5, help='save this many fail examples (pickle)')
    return p.parse_args()

def main():
    args = parse_args()
    f = args.f
    trials = args.trials
    processes = args.processes
    L = args.L; r = args.r; h = args.h; thresh = args.thresh
    seed_base = args.seed_base

    N = compute_N_from_f(f, L, r, h)
    print(f"High-precision verify: f={f} -> N={N}, trials={trials}, processes={processes}")

    # prepare output dirs
    out_dir = os.path.join('results','outputs')
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # prepare args list
    args_list=[]
    for t in range(trials):
        seed = int(seed_base + t + (int(abs(f)*1e9) % 1000000))
        args_list.append((seed, N, L, r, h, thresh))

    # run (sequential or parallel)
    results=[]
    start_all = time.time()
    if processes == 1:
        for a in args_list:
            results.append(trial_worker(a))
    else:
        # careful: use if __name__ guard if this script imported; but when run directly it's OK
        with Pool(processes) as pool:
            for res in pool.imap_unordered(trial_worker, args_list):
                results.append(res)
    total_time = time.time() - start_all

    # collect summary
    successes = sum(1 for r in results if r['connected'])
    p_hat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)

    ts = datetime.utcnow().isoformat() + 'Z'
    summary = {
        'f': f, 'N': N, 'trials': trials, 'successes': successes, 'p_hat': p_hat, 'ci_low': ci_low, 'ci_high': ci_high, 'elapsed_s': total_time, 'timestamp': ts
    }
    # save summary
    f_str = f"{f:.6f}".replace('.', 'p')
    summary_path = os.path.join(out_dir, f"high_precision_f{f_str}_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print("Summary saved to", summary_path)
    # save per-trial CSV (without segments)
    trials_path = os.path.join(out_dir, f"high_precision_f{f_str}_trials.csv")
    df_trials = pd.DataFrame([{'seed': r['seed'], 'connected': int(r['connected']), 'elapsed_s': r['elapsed_s']} for r in results])
    df_trials.to_csv(trials_path, index=False)
    print("Per-trial saved to", trials_path)

    # save sample pickles: keep first K successes and first K failures
    succ_saved=0; fail_saved=0
    for r in results:
        if r['connected'] and succ_saved < args.save_success:
            # pick a sample to save: need segments; but trial_worker returns None for segments when not saved to reduce memory
            # here we reconstructed segments in worker and returned them only if connected? we returned None earlier in template.
            # If segments field is None, re-generate sample by seed
            segments = r.get('segments')
            if segments is None:
                # regenerate
                cyls = sample_random_cylinders(N, L, r, h, seed=r['seed'])
                segs=[]
                for c in cyls:
                    segs.extend(split_cylinder_by_box(c, L))
                segments = segs
            pkl_path = os.path.join(samples_dir, f"f{f_str}_succ_seed{r['seed']}.pkl")
            with open(pkl_path, 'wb') as fout:
                pickle.dump({'seed': r['seed'], 'segments': segments, 'f': f}, fout)
            succ_saved += 1
        if (not r['connected']) and fail_saved < args.save_fail:
            segments = r.get('segments')
            if segments is None:
                cyls = sample_random_cylinders(N, L, r, h, seed=r['seed'])
                segs=[]
                for c in cyls:
                    segs.extend(split_cylinder_by_box(c, L))
                segments = segs
            pkl_path = os.path.join(samples_dir, f"f{f_str}_fail_seed{r['seed']}.pkl")
            with open(pkl_path, 'wb') as fout:
                pickle.dump({'seed': r['seed'], 'segments': segments, 'f': f}, fout)
            fail_saved += 1
        if succ_saved >= args.save_success and fail_saved >= args.save_fail:
            break

    print(f"Saved {succ_saved} success and {fail_saved} fail sample(s) to {samples_dir}")
    print("Done.")

if __name__ == '__main__':
    main()