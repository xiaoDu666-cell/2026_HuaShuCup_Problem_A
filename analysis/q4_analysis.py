# analysis/q4_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path
import time
import matplotlib

# ============================================================
# 设置中文字体
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
data_dir = Path("results/outputs")
fig_dir = Path("results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

def find_file(base_dir, basename):
    for ext in ['.csv', '.xlsx', '.xls']:
        path = base_dir / f"{basename}{ext}"
        if path.exists():
            return path
    return None

def load_data(filepath):
    if filepath is None:
        return pd.DataFrame()
    ext = filepath.suffix.lower()
    if ext == '.csv':
        return pd.read_csv(filepath)
    else:
        return pd.read_excel(filepath)

coarse_path = find_file(data_dir, "q4_coarse_results")
fine_path = find_file(data_dir, "q4_fine_results")
final_csv = data_dir / "q4_best_answer.csv"

print(f"查找数据文件:")
print(f"  coarse: {coarse_path} (exists: {coarse_path is not None})")
print(f"  fine:   {fine_path} (exists: {fine_path is not None})")

df_coarse = load_data(coarse_path)
df_fine = load_data(fine_path)
df_all = pd.concat([df_coarse, df_fine], ignore_index=True) if (not df_coarse.empty or not df_fine.empty) else pd.DataFrame()

print(f"Loaded coarse rows={len(df_coarse)}, fine rows={len(df_fine)}, total rows={len(df_all)}")

# --- 2) select best point ---
def select_best_and_save(df_all, threshold=0.90, out_path=final_csv):
    if df_all.empty:
        print("No data to select from.")
        return None
    for c in ['p_hat','ci_low','cost','fA','fB']:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
    feasible = df_all[df_all['p_hat'] >= threshold].copy()
    source = None
    if feasible.empty:
        feasible = df_all[df_all['ci_low'] >= threshold].copy()
        if feasible.empty:
            print("No feasible point found.")
            df_all['score'] = df_all['p_hat'] / (df_all['cost'] + 1e-12)
            cand = df_all.sort_values('score', ascending=False).iloc[0]
            cand = cand.to_dict()
            cand['found'] = False
            cand['note'] = 'No feasible point; suggested best by p_hat/cost ratio'
            pd.DataFrame([cand]).to_csv(out_path, index=False)
            return cand
        else:
            source = 'ci_low'
    else:
        source = 'p_hat'
    best = feasible.loc[feasible['cost'].idxmin()]
    best = best.to_dict()
    best['found'] = True
    best['selection_source'] = source
    best['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pd.DataFrame([best]).to_csv(out_path, index=False)
    print("Saved best answer to", out_path)
    return best

best = select_best_and_save(df_all, threshold=0.9, out_path=final_csv)
print("Best:", best)

# --- 3) plotting ---
plot_dir = fig_dir

def plot_probability_heatmap(df_all, out_png=plot_dir/"q4_pheatmap.png"):
    if df_all.empty:
        print("No data for heatmap")
        return
    x = df_all['fB'].values * 100.0
    y = df_all['fA'].values * 100.0
    z = df_all['p_hat'].values
    triang = tri.Triangulation(x, y)
    plt.figure(figsize=(7,6))
    cf = plt.tricontourf(triang, z, levels=20, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(cf, label='p_hat')
    plt.scatter(x, y, c='k', s=6, alpha=0.4)
    # 自动裁剪 x/y 轴范围到数据范围 + 边距
    x_margin = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 0.01
    y_margin = (y.max() - y.min()) * 0.05 if y.max() > y.min() else 0.01
    plt.xlim(x.min() - x_margin, x.max() + x_margin)
    plt.ylim(y.min() - y_margin, y.max() + y_margin)
    plt.xlabel('fB (%)'); plt.ylabel('fA (%)')
    plt.title('导通概率热力图 p_hat (fA vs fB)')
    if best is not None and best.get('found', False):
        plt.plot(best['fB']*100.0, best['fA']*100.0, 'r*', markersize=14, label='best')
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved", out_png)

def plot_cost_contours_with_feasible(df_all, out_png=plot_dir/"q4_cost_contour_feasible.png"):
    if df_all.empty:
        print("No data for cost contour")
        return
    x = df_all['fB'].values * 100.0
    y = df_all['fA'].values * 100.0
    z = df_all['cost'].values
    p = df_all['p_hat'].values
    triang = tri.Triangulation(x, y)
    plt.figure(figsize=(7,6))
    levels = 12
    cs = plt.tricontourf(triang, z, levels=levels, cmap='plasma')
    plt.colorbar(cs, label='cost (元)')
    mask = p >= 0.9
    if mask.any():
        plt.scatter(x[mask], y[mask], facecolors='none', edgecolors='k', label='feasible (p>=0.9)')
    if best is not None and best.get('found', False):
        plt.plot(best['fB']*100.0, best['fA']*100.0, 'r*', markersize=14, label='best')
    # 自动裁剪
    x_margin = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 0.01
    y_margin = (y.max() - y.min()) * 0.05 if y.max() > y.min() else 0.01
    plt.xlim(x.min() - x_margin, x.max() + x_margin)
    plt.ylim(y.min() - y_margin, y.max() + y_margin)
    plt.xlabel('fB (%)'); plt.ylabel('fA (%)')
    plt.title('成本等高线 (可行域标注)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved", out_png)

def plot_slices(df_all, fixed_fB_list=None, out_png=plot_dir/"q4_slices.png"):
    if df_all.empty:
        print("No data for slices")
        return
    if fixed_fB_list is None:
        fB_vals = np.unique(np.round(df_all['fB'].values, 8))
        if len(fB_vals) > 5:
            fixed_fB_list = [fB_vals[0], fB_vals[len(fB_vals)//2], fB_vals[-1]]
        else:
            fixed_fB_list = fB_vals
    plt.figure(figsize=(8,5))
    for fB in fixed_fB_list:
        sel = df_all[np.isclose(df_all['fB'], fB, atol=1e-12)]
        if sel.empty:
            sel = df_all[np.isclose(df_all['fB'], fB, atol=1e-4)]
        if sel.empty:
            continue
        sel = sel.sort_values('fA')
        plt.plot(sel['fA']*100.0, sel['p_hat'], marker='o', label=f"fB={fB*100:.3f}%")
    plt.axhline(0.9, color='r', linestyle='--', label='threshold=0.9')
    plt.xlabel('fA (%)'); plt.ylabel('p_hat')
    plt.title('固定 fB 的 p_hat 随 fA 变化')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved", out_png)

def plot_pareto(df_all, out_png=plot_dir/"q4_pareto.png"):
    if df_all.empty:
        return
    df = df_all.copy()
    plt.figure(figsize=(6,5))
    plt.scatter(df['p_hat'], df['cost'], c='C0', alpha=0.6)
    plt.xlabel('p_hat')
    plt.ylabel('cost (元)')
    plt.title('Pareto plot: cost vs p_hat')
    feas = df[df['p_hat']>=0.9]
    if not feas.empty:
        plt.scatter(feas['p_hat'], feas['cost'], c='red', label='feasible (p>=0.9)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved", out_png)

# run plotting
print("\n开始生成图表...")
plot_probability_heatmap(df_all)
plot_cost_contours_with_feasible(df_all)
plot_slices(df_all)
plot_pareto(df_all)

print("\nAll done.")
print(f"图片已保存至: {fig_dir.absolute()}")
print(f"最优解已保存至: {final_csv.absolute()}")

# --- 打印最优解 ---
if best is not None and best.get('found', False):
    print("\n" + "="*50)
    print("问题四最优解")
    print("="*50)
    print(f"fA = {best['fA']:.6f} ({best['fA']*100:.4f}%)")
    print(f"fB = {best['fB']:.6f} ({best['fB']*100:.4f}%)")
    print(f"N_A = {best['N_A']:.0f} 根")
    print(f"N_B = {best['N_B']:.0f} 个")
    print(f"p_hat = {best['p_hat']:.4f} ({best['p_hat']*100:.2f}%)")
    print(f"95% CI = [{best['ci_low']:.4f}, {best['ci_high']:.4f}]")
    print(f"成本 = {best['cost']:.4f} 元")
    print("="*50)