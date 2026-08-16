#!/usr/bin/env python3
"""
scripts/run_monte_carlo_B.py

单介质（B相球）蒙特卡洛扫描，找导通概率达到 100%（或目标值）的临界体积分数。
保存结果到 results/outputs/monte_carlo_B_results.csv。
"""

import os
import sys
import time
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd

# ------------------ 参数（在脚本顶部集中修改） ------------------
L = 10000.0        # nm, 盒子边长
r = 200.0          # nm, B相球半径
thresh = 1.8       # nm, 表面到表面认为连通的阈值

# Monte Carlo 参数
#f_grid = np.linspace(0.000, 0.015, 31)   # 0 ~ 1.5%，步长0.0005（0.05%）
f_grid = np.array([0.010, 0.015, 0.020, 0.025])  # 1.0%, 1.5%, 2.0%, 2.5%

trials_per_f = 500                        # 先用500次粗扫，找到上界后再精扫
processes = 12                            # 使用 12 核并行
seed_base = 20260812

# 输出路径
OUT_DIR = os.path.join('results', 'outputs')
OUT_CSV = os.path.join(OUT_DIR, 'monte_carlo_B_results.csv')
# ----------------------------------------------------------------

# 把 repo 根目录加入 sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root_candidate = os.path.abspath(os.path.join(script_dir, '..'))
if repo_root_candidate not in sys.path:
    sys.path.insert(0, repo_root_candidate)

# 导入项目内模块
try:
    from src.geometry import Sphere
    from src.truncation import split_sphere_by_box
    from src.connectivity import build_connectivity
except Exception as e:
    print("Error importing project modules from src/:", e)
    raise

# ------------------ 帮助函数 ------------------
def volume_sphere(rad):
    return (4.0/3.0) * np.pi * (rad ** 3)

def compute_N_from_f(f, Lbox, rad):
    V_box = Lbox ** 3
    V_sph = volume_sphere(rad)
    if V_sph <= 0:
        raise ValueError("Sphere volume must be positive")
    N = int(round((f * V_box) / V_sph))
    return max(0, N)

def sample_random_spheres(N, Lbox, rad, seed=None):
    """随机生成 N 个球，返回 Sphere 列表。
    中心均匀分布在 [-L/2, L/2]^3。
    """
    if N <= 0:
        return []
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-Lbox/2, Lbox/2, size=(N, 3))
    spheres = []
    for i in range(N):
        spheres.append(Sphere(centers[i], rad, id=i))
    return spheres

def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + (z**2) / n
    centre = phat + (z**2) / (2 * n)
    adj = z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

def run_single_trial(args):
    """单次试验，返回 1（连通）或 0（不连通）。
    args: (N, L, r, thresh, seed)
    """
    N, Lbox, rad, thr, seed = args
    spheres = sample_random_spheres(N, Lbox, rad, seed=seed)
    segments = []
    for s in spheres:
        segs = split_sphere_by_box(s, Lbox)
        segments.extend(segs)

    # 打印类型统计
    sphere_count = sum(1 for s in segments if isinstance(s, Sphere))
    cap_count = sum(1 for s in segments if hasattr(s, 'n') and hasattr(s, 'd'))
    #print(f"segments: Sphere={sphere_count}, SphericalCap={cap_count}")
    connected, _uf = build_connectivity(segments, Lbox, thr)
    return 1 if connected else 0

def run_monte_carlo_for_f(f, trials, Lbox, rad, thr, seed_base=0, processes=1):
    N = compute_N_from_f(f, Lbox, rad)
    if trials <= 0 or N == 0:
        return {
            'f': float(f), 'N': int(N), 'trials': int(trials),
            'successes': 0, 'p_hat': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'elapsed_s': 0.0
        }
    args_list = []
    for t in range(trials):
        seed = int(seed_base + t + (int(abs(f) * 1e9) % 1000000))
        args_list.append((N, Lbox, rad, thr, seed))

    t0 = time.time()
    if processes == 1:
        results = [run_single_trial(a) for a in args_list]
    else:
        with Pool(processes) as pool:
            results = pool.map(run_single_trial, args_list)
    successes = int(sum(results))
    p_hat = successes / trials
    ci_low, ci_high = wilson_interval(successes, trials)
    elapsed = time.time() - t0
    return {
        'f': float(f), 'N': int(N), 'trials': int(trials),
        'successes': successes, 'p_hat': float(p_hat),
        'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'elapsed_s': float(elapsed)
    }

# ------------------ 主流程 ------------------
def main():
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = []
    overall_t0 = time.time()
    print(f"B相单填料扫描: f_grid={len(f_grid)}点, trials_per_f={trials_per_f}, processes={processes}")
    print(f"f范围: {f_grid[0]:.4f} ~ {f_grid[-1]:.4f}")
    print("-" * 60)

    for f in f_grid:
        print(f"\n正在扫 f = {f:.5f}", end=" ", flush=True)
        res = run_monte_carlo_for_f(
            f,
            trials_per_f,
            L,
            r,
            thresh,
            seed_base=seed_base,
            processes=processes
        )
        status = "✅" if res['p_hat'] >= 0.99 else "⏳"
        print(f"{status} N={res['N']}, p={res['p_hat']:.4f} ({res['ci_low']:.4f}, {res['ci_high']:.4f}), time={res['elapsed_s']:.1f}s")
        all_results.append(res)

        # 每步保存
        df_tmp = pd.DataFrame(all_results)
        df_tmp.to_csv(OUT_CSV, index=False)

    overall_elapsed = time.time() - overall_t0
    print(f"\n✅ 全部完成! 总耗时 {overall_elapsed:.1f}s")
    print(f"结果保存至 {OUT_CSV}")

    # 找第一个 p_hat >= 1.0 的点（即100%导通）
    df = pd.DataFrame(all_results)
    p100 = df[df['p_hat'] >= 1.0]
    if not p100.empty:
        critical = p100.iloc[0]
        print(f"\n🎯 首次达到100%导通: f = {critical['f']:.5f}, N = {int(critical['N'])}")
        print(f"   p_hat = {critical['p_hat']:.4f}, 置信区间 ({critical['ci_low']:.4f}, {critical['ci_high']:.4f})")
    else:
        print(f"\n⚠️ 在 f_max={f_grid[-1]:.4f} 范围内未达到100%导通")

if __name__ == '__main__':
    main()