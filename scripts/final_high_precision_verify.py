#!/usr/bin/env python3
"""
Final high-precision verification script.

Save as: scripts/final_high_precision_verify.py

Examples:
  # Linux / macOS
  python scripts/final_high_precision_verify.py --f 0.001 --trials 5000 --processes 8 --save_samples 3

  # Windows PowerShell
  python .\scripts\final_high_precision_verify.py --f 0.001 --trials 5000 --processes 4 --save_samples 3

Outputs:
  - results/outputs/final_verify_summary_{timestamp}.csv
  - results/outputs/final_verify_summary_{timestamp}.json
  - results/samples/seed_{seed}.pkl  (for saved successful trials)
  - results/samples/seed_{seed}.png  (XY projection of that sample)
"""
import os
import sys
import time
import math
import argparse
import json
import pickle
from datetime import datetime
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd

# try to import scipy for Clopper-Pearson interval; if not available, we'll skip CP
try:
    from scipy.stats import beta
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ensure repo root in sys.path if script is in scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# import project modules
try:
    from src.geometry import Cylinder
    from src.truncation import split_cylinder_by_box
    from src.connectivity import build_connectivity
    from src.geometry import segment_plane_distance_to_x_plane, cylinder_surface_distance
except Exception as e:
    print("Error importing project modules from src/:", e)
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
    denom = 1 + (z**2)/n
    centre = phat + (z**2)/(2*n)
    adj = z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

def clopper_pearson_interval(k, n, alpha=0.05):
    """Return (low, high) Clopper-Pearson two-sided interval; requires scipy if available."""
    if not _HAS_SCIPY:
        return None, None
    if n == 0:
        return 0.0, 1.0
    if k == 0:
        low = 0.0
    else:
        low = beta.ppf(alpha/2, k, n - k + 1)
    if k == n:
        high = 1.0
    else:
        high = beta.ppf(1 - alpha/2, k + 1, n - k)
    return float(low), float(high)

# run single trial (module-level for pickling)
def _run_trial(args):
    N, L, r, h, thresh, seed = args
    cyls = sample_random_cylinders(N, L, r, h, seed=seed)
    segments = []
    for c in cyls:
        segments.extend(split_cylinder_by_box(c, L))
    connected, _ = build_connectivity(segments, L, thresh)
    return connected  # 1 or 0 (bool -> int)

def parse_args():
    p = argparse.ArgumentParser(description='Final high-precision Monte Carlo verification for p(f).')
    p.add_argument('--f', type=float, required=True, help='volume fraction to verify (decimal, e.g., 0.001)')
    p.add_argument('--trials', type=int, default=5000, help='number of trials (default 5000)')
    p.add_argument('--processes', type=int, default=4, help='number of parallel worker processes')
    p.add_argument('--seed', type=int, default=20260812, help='seed base')
    p.add_argument('--L', type=float, default=10000.0, help='box size (nm)')
    p.add_argument('--r', type=float, default=30.0, help='cylinder radius (nm)')
    p.add_argument('--h', type=float, default=5000.0, help='cylinder length (nm)')
    p.add_argument('--thresh', type=float, default=1.8, help='surface-to-surface threshold (nm)')
    p.add_argument('--out_dir', type=str, default=os.path.join('results','outputs'), help='output directory')
    p.add_argument('--save_samples', type=int, default=3, help='save up to this many successful trial samples (pickle+png)')
    p.add_argument('--no-cp', action='store_true', help='skip Clopper–Pearson even if scipy available')
    return p.parse_args()

def save_sample(seed, f, L, r, h, thresh, out_sample_dir, sample_idx):
    """Regenerate sample for given seed, save segments as pickle and an XY PNG projection."""
    N = compute_N_from_f(f, L, r, h)
    cyls = sample_random_cylinders(N, L, r, h, seed=seed)
    segments = []
    for c in cyls:
        segments.extend(split_cylinder_by_box(c, L))
    # Build connectivity to extract the connected cluster if connected
    connected, uf = build_connectivity(segments, L, thresh)
    # Save segments structure
    seg_serial = []
    for s in segments:
        seg_serial.append({'p0': s.p0.tolist(), 'p1': s.p1.tolist(), 'r': float(s.r), 'id': int(s.id)})
    os.makedirs(out_sample_dir, exist_ok=True)
    fname = os.path.join(out_sample_dir, f"sample_seed_{seed}.pkl")
    with open(fname, 'wb') as fout:
        pickle.dump({'seed': int(seed), 'f': f, 'N': N, 'segments': seg_serial, 'connected': bool(connected)}, fout)
    # create XY scatter of segments centers colored by whether they belong to cluster that spans (if possible)
    centers = np.array([ (np.array(s['p0']) + np.array(s['p1']))/2.0 for s in seg_serial ])
    # Try to find connected cluster root (we can reuse uf to find roots if available)
    cluster_mask = None
    try:
        # build root mapping
        roots = [uf.find(i) for i in range(len(segments))]
        # find roots that touch left and right
        left_roots=set(); right_roots=set()
        for i,s in enumerate(segments):
            d_left = segment_plane_distance_to_x_plane(s.p0, s.p1, -L/2.0); surf_left = max(0.0,d_left - s.r)
            if surf_left <= thresh + 1e-9:
                left_roots.add(roots[i])
            d_right = segment_plane_distance_to_x_plane(s.p0, s.p1, L/2.0); surf_right = max(0.0,d_right - s.r)
            if surf_right <= thresh + 1e-9:
                right_roots.add(roots[i])
        spanning_roots = left_roots & right_roots
        if spanning_roots:
            target_root = next(iter(spanning_roots))
            # mark segments with same root
            cluster_mask = np.array([ (r == target_root) for r in roots ], dtype=bool)
    except Exception:
        cluster_mask = None

    # Plot XY
    plt.figure(figsize=(8,8))
    if cluster_mask is None:
        plt.scatter(centers[:,0], centers[:,1], s=6, color='blue', alpha=0.5)
    else:
        plt.scatter(centers[~cluster_mask,0], centers[~cluster_mask,1], s=6, color='blue', alpha=0.4, label='other')
        plt.scatter(centers[cluster_mask,0], centers[cluster_mask,1], s=10, color='red', alpha=0.9, label='spanning cluster')
        plt.legend()
    plt.axvline(-L/2.0, color='k', linestyle='--'); plt.axvline(L/2.0, color='k', linestyle='--')
    plt.gca().set_aspect('equal', 'box')
    plt.title(f"Sample seed={seed}, f={f*100:.4f}%")
    png_fname = os.path.join(out_sample_dir, f"sample_seed_{seed}.png")
    plt.savefig(png_fname, dpi=200, bbox_inches='tight')
    plt.close()
    return fname, png_fname

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # summary paths
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    summary_csv = os.path.join(args.out_dir, f"final_verify_summary_{ts}.csv")
    summary_json = os.path.join(args.out_dir, f"final_verify_summary_{ts}.json")
    sample_dir = os.path.join('results','samples')

    # compute N for info
    N = compute_N_from_f(args.f, args.L, args.r, args.h)
    print(f"Final verification: f={args.f} -> N={N}, trials={args.trials}, processes={args.processes}")

    # build args list for trials (deterministic seeds so we can regenerate samples)
    trials = int(args.trials)
    args_list = []
    seeds = []
    for t in range(trials):
        seed = int(args.seed + t + (int(abs(args.f) * 1e9) % 1000000))
        seeds.append(seed)
        args_list.append((N, args.L, args.r, args.h, args.thresh, seed))

    # run trials
    t0 = time.time()
    if args.processes == 1:
        results = [ _run_trial(a) for a in args_list ]
    else:
        try:
            multiprocessing.set_start_method('spawn', force=False)
        except RuntimeError:
            pass
        with Pool(args.processes) as pool:
            results = pool.map(_run_trial, args_list)
    elapsed = time.time() - t0

    successes = int(sum(results))
    p_hat = successes / trials
    wilson_low, wilson_high = wilson_interval(successes, trials)
    cp_low, cp_high = (None, None)
    if _HAS_SCIPY and not args.no_cp:
        try:
            cp_low, cp_high = clopper_pearson_interval(successes, trials, alpha=0.05)
        except Exception:
            cp_low, cp_high = (None, None)

    # Save summary
    summary = {
        'timestamp_utc': datetime.utcnow().isoformat()+'Z',
        'f': args.f,
        'N': N,
        'trials': trials,
        'successes': successes,
        'p_hat': p_hat,
        'wilson_low_95': wilson_low,
        'wilson_high_95': wilson_high,
        'cp_low_95': cp_low,
        'cp_high_95': cp_high,
        'elapsed_s': elapsed,
        'processes': args.processes,
        'seed_base': args.seed,
        'L': args.L,
        'r': args.r,
        'h': args.h,
        'thresh': args.thresh
    }
    df = pd.DataFrame([summary])
    df.to_csv(summary_csv, index=False)
    with open(summary_json, 'w') as fo:
        json.dump(summary, fo, indent=2)
    print("Summary saved to:", summary_csv, summary_json)
    print("Result: successes={}/{} p_hat={:.6f} Wilson95=({:.4f},{:.4f}) CP95=({}, {})".format(
        successes, trials, p_hat, wilson_low, wilson_high, cp_low, cp_high))

    # Save some successful trial samples (regenerate main-process to avoid heavy IPC)
    saved_samples = []
    if successes > 0 and args.save_samples > 0:
        os.makedirs(sample_dir, exist_ok=True)
        count = 0
        for idx, flag in enumerate(results):
            if flag:
                seed = seeds[idx]
                print("Saving sample for seed", seed)
                try:
                    pkl_path, png_path = save_sample(seed, args.f, args.L, args.r, args.h, args.thresh, sample_dir, count)
                    saved_samples.append({'seed': seed, 'pkl': pkl_path, 'png': png_path})
                    count += 1
                    if count >= args.save_samples:
                        break
                except Exception as e:
                    print("Warning: failed to save sample for seed", seed, "error:", e)
    # write saved_samples into summary JSON as well
    summary['saved_samples'] = saved_samples
    with open(summary_json, 'w') as fo:
        json.dump(summary, fo, indent=2)

    print("Done. Saved {} sample(s).".format(len(saved_samples)))
    print("You can re-generate visualizations from the .pkl files in results/samples/")

if __name__ == '__main__':
    main()