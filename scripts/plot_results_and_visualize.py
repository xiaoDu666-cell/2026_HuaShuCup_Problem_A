#!/usr/bin/env python3
"""
Plot p(f) curve, logistic fit, cluster fraction and visualize saved samples.

Usage:
  python scripts/plot_results_and_visualize.py --mc_csv results/outputs/monte_carlo_pf.csv \
      --high_csv results/outputs/high_precision_f0p001_summary.csv --samples_dir results/outputs/samples --out_dir figures
"""
import os, sys, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ensure repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# for sample visualization
import pickle
from mpl_toolkits.mplot3d import Axes3D  # noqa

def logistic(x, a, b):
    # logistic: p = 1 / (1 + exp(-(a + b*x)))
    return 1.0 / (1.0 + np.exp(-(a + b * x)))

def fit_logistic(fs, phats, weights=None):
    # initial guess
    p0 = [0.0, 1.0]
    popt, pcov = curve_fit(logistic, fs, phats, p0=p0, sigma=weights, maxfev=10000)
    return popt, pcov

def plot_pf_with_ci(mc_df, out_dir, high_res=True):
    os.makedirs(out_dir, exist_ok=True)
    fs = mc_df['f'].values
    p_hat = mc_df['p_hat'].values
    ci_low = mc_df['ci_low'].values
    ci_high = mc_df['ci_high'].values
    # convert to percent
    x = fs * 100.0
    plt.figure(figsize=(7,5), dpi=300 if high_res else 100)
    plt.errorbar(x, p_hat, yerr=[p_hat - ci_low, ci_high - p_hat], fmt='o-', capsize=4, markersize=6, label='p_hat (95% CI)')
    plt.axhline(0.9, color='gray', linestyle='--', label='p=0.9')
    # logistic fit on f (not percent)
    try:
        popt, pcov = fit_logistic(fs, p_hat)
        xs = np.linspace(fs.min(), fs.max(), 200)
        ys = logistic(xs, *popt)
        plt.plot(xs*100, ys, '-', color='C1', label='logistic fit')
        # find f at p=0.9
        # solve logistic(f) = 0.9 -> a + b f = -ln(1/0.9 - 1)
        a, b = popt
        val = -math.log(1.0/0.9 - 1.0)
        f_at_09 = (val - a) / b
        plt.axvline(f_at_09*100, color='C1', linestyle=':', label=f"f@p=0.9 ~ {f_at_09*100:.3f}%")
        # save fit details
        fittxt = os.path.join(out_dir, "pf_logistic_fit.txt")
        with open(fittxt, 'w') as f:
            f.write(f"logistic params a={a}, b={b}\n")
            f.write(f"f@p=0.5 = {-a/b if b!=0 else float('nan')}\n")
            f.write(f"f@p=0.9 = {f_at_09}\n")
    except Exception as e:
        print("Logistic fit failed:", e)
    plt.xlabel("体积分数 f (%)")
    plt.ylabel("导通概率 p(f)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    outpath = os.path.join(out_dir, 'pf_curve.svg')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.savefig(outpath.replace('.svg','.png'), dpi=300)
    plt.show()
    print("Saved pf plot to", outpath)

def visualize_sample_pickle(pkl_path, out_dir, figsize=(8,8)):
    os.makedirs(out_dir, exist_ok=True)
    with open(pkl_path, 'rb') as fin:
        obj = pickle.load(fin)
    segments = obj.get('segments', None)
    seed = obj.get('seed', None)
    fval = obj.get('f', None)
    if segments is None:
        print("No segments in", pkl_path)
        return
    # get centers and whether touching left/right
    centers = np.array([seg.center() for seg in segments])
    # simple XY scatter: color segments by whether their root is in connected cluster? we don't have uf here
    # color by x position cluster: segments near left/right plane markers
    xs = centers[:,0]; ys = centers[:,1]; zs = centers[:,2]
    plt.figure(figsize=figsize)
    plt.scatter(xs, ys, s=6, c='blue', alpha=0.6)
    plt.xlabel('X (nm)'); plt.ylabel('Y (nm)')
    plt.title(f'XY projection seed={seed} f={fval}')
    plt.axvline(-5000, color='k', linestyle='--'); plt.axvline(5000, color='k', linestyle='--')
    outpng = os.path.join(out_dir, f"sample_xy_seed{seed}.png")
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.show()
    # 3D
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, '.', markersize=4, alpha=0.6, color='C0')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'3D sample seed={seed}')
    out3d = os.path.join(out_dir, f"sample_3d_seed{seed}.png")
    plt.tight_layout()
    plt.savefig(out3d, dpi=300)
    plt.show()
    print("Saved sample visuals to", outdir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mc_csv', type=str, required=True, help='Monte Carlo CSV with f,p_hat,ci_low,ci_high')
    p.add_argument('--high_csv', type=str, default=None, help='High-precision summary CSV (optional)')
    p.add_argument('--samples_dir', type=str, default='results/outputs/samples', help='Directory with sample pickles')
    p.add_argument('--out_dir', type=str, default='figures', help='Output dir for figures')
    args = p.parse_args()

    mc_df = pd.read_csv(args.mc_csv)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    plot_pf_with_ci(mc_df, out_dir)

    # if sample pickles exist, visualize a few
    if os.path.isdir(args.samples_dir):
        files = sorted([f for f in os.listdir(args.samples_dir) if f.endswith('.pkl')])
        if files:
            # visualize up to 4 samples
            to_vis = files[:4]
            for fn in to_vis:
                pkl_path = os.path.join(args.samples_dir, fn)
                sample_out = os.path.join(out_dir, 'samples')
                visualize_sample_pickle(pkl_path, sample_out)
        else:
            print("No sample pickles found in", args.samples_dir)
    else:
        print("Samples dir not found:", args.samples_dir)

if __name__ == '__main__':
    main()