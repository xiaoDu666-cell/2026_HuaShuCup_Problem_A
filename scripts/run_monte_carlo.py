#!/usr/bin/env python3
"""
scripts/run_monte_carlo.py

独立运行的蒙特卡洛脚本（单介质导通概率）。
保存结果到 results/outputs/monte_carlo_results.csv。

依赖：
- Python 3.8+
- numpy, pandas, matplotlib (可选)
- 项目内 src/geometry.py, src/truncation.py, src/connectivity.py（并保证能通过 sys.path 导入）
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
r = 30.0           # nm, 圆柱半径
h = 5000.0         # nm, 圆柱长度
thresh = 1.8       # nm, 表面到表面认为连通的阈值

# Monte Carlo 参数
f_grid = np.linspace(0.001, 0.02, 20)   # 体积分数网格（例如 0.1% ~ 2.0%）
trials_per_f = 500                       # 每个 f 的独立试验次数
processes = 4                            # 并行进程数（根据机器调整）
seed_base = 20260812                     # 随机种子基数

# 输出路径（相对于当前工作目录或脚本位置）
OUT_DIR = os.path.join('results', 'outputs')
OUT_CSV = os.path.join(OUT_DIR, 'monte_carlo_results.csv')
# ----------------------------------------------------------------

# 尝试把 repo 根目录加入 sys.path，以便导入 src 包（假设脚本位于 repo/scripts/ 下）
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root_candidate = os.path.abspath(os.path.join(script_dir, '..'))
if repo_root_candidate not in sys.path:
    sys.path.insert(0, repo_root_candidate)

# 导入项目内模块（确保 src 中有相应实现）
try:
    from src.geometry import Cylinder
    from src.truncation import split_cylinder_by_box
    from src.connectivity import build_connectivity
except Exception as e:
    print("Error importing project modules from src/:", e)
    print("Make sure this script is placed under the repository and src/ is importable.")
    raise

# ------------------ 帮助函数 ------------------
def volume_cylinder(rad, length):
    return np.pi * (rad ** 2) * length

def compute_N_from_f(f, Lbox, rad, length):
    V_box = Lbox ** 3
    V_cyl = volume_cylinder(rad, length)
    if V_cyl <= 0:
        raise ValueError("Cylinder volume must be positive")
    N = int(round((f * V_box) / V_cyl))
    return max(0, N)

def sample_random_cylinders(N, Lbox, rad, length, seed=None):
    """随机生成 N 个圆柱，返回 Cylinder 列表。
    中心均匀分布在 [-L/2, L/2]^3，方向均匀球面分布，长度固定为 length。
    """
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
    adj = z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

def run_single_trial(args):
    """单次试验，返回 1（连通）或 0（不连通）。
    args: (N, L, r, h, thresh, seed)
    """
    N, Lbox, rad, length, thr, seed = args
    cyls = sample_random_cylinders(N, Lbox, rad, length, seed=seed)
    segments = []
    for c in cyls:
        segs = split_cylinder_by_box(c, Lbox)
        # split_cylinder_by_box 应返回盒内片段列表（可能为空）
        segments.extend(segs)
    connected, _uf = build_connectivity(segments, Lbox, thr)
    return 1 if connected else 0

def run_monte_carlo_for_f(f, trials, Lbox, rad, length, thr, seed_base=0, processes=1):
    N = compute_N_from_f(f, Lbox, rad, length)
    if trials <= 0 or N == 0:
        return {
            'f': float(f), 'N': int(N), 'trials': int(trials),
            'successes': 0, 'p_hat': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'elapsed_s': 0.0
        }
    args_list = []
    for t in range(trials):
        # 不同试验使用不同 seed，加入 f 的信息以避免不同 f 下 seed 相同
        seed = int(seed_base + t + (int(abs(f) * 1e9) % 1000000))
        args_list.append((N, Lbox, rad, length, thr, seed))

    t0 = time.time()
    if processes == 1:
        results = [run_single_trial(a) for a in args_list]
    else:
        # Windows: spawn-start 方法需要在 if __name__ guard 中设置；这里外层会处理
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
    # 尝试设置多进程启动方式（Windows 建议使用 'spawn'）
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        # start method 已设置，忽略
        pass
    # 输出目录
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = []
    overall_t0 = time.time()
    print(f"Starting Monte Carlo: f_grid len={len(f_grid)}, trials_per_f={trials_per_f}, processes={processes}")
    for f in f_grid:
        print(f"\nRunning f = {f:.6f}")
        res = run_monte_carlo_for_f(
            f,
            trials_per_f,
            L,
            r,
            h,
            thresh,
            seed_base=seed_base,
            processes=processes
        )
        print(f"  N={res['N']}, successes={res['successes']}/{res['trials']}, p_hat={res['p_hat']:.4f}, time={res['elapsed_s']:.1f}s")
        all_results.append(res)

        # 每次写入中间结果（便于意外终止后恢复）
        try:
            df_tmp = pd.DataFrame(all_results)
            df_tmp.to_csv(OUT_CSV, index=False)
        except Exception as e:
            print("Warning: failed to write interim CSV:", e)

    overall_elapsed = time.time() - overall_t0
    print(f"\nAll done in {overall_elapsed:.1f}s. Results saved to {OUT_CSV}")

    # 最终保存
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)


if __name__ == '__main__':
    # 在 Windows 中，multiprocessing 要求可导入的模块是顶层可导入的，因此要保证此处被保护
    main()