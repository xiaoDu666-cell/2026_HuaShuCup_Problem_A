import numpy as np
from src.geometry import Sphere
from src.truncation import split_sphere_by_box
from src.connectivity import build_connectivity

L = 10000.0
r = 200.0
thresh = 1.8
half = L / 2.0

print("=" * 70)
print("验证介质B（球）的边界截断和连通性")
print("=" * 70)

# ============================================================
# 场景1：一排球完全在盒内，从左电极到右电极
# ============================================================
print("\n【场景1】一排球从左电极到右电极（完全在盒内）")
print("-" * 70)

# 球心沿 X 轴排列，间距 300nm（保证连通，因为 2*r + thresh = 401.8nm）
# 第一个球接触左电极：球心在 -5000 + 200 = -4800
# 最后一个球接触右电极：球心在 5000 - 200 = 4800
# 从 -4800 到 4800，间距 300，需要 33 个球
x_positions = np.linspace(-4800, 4800, 33)
spheres = []
for i, x in enumerate(x_positions):
    spheres.append(Sphere(np.array([x, 0.0, 0.0]), r, id=i))

print(f"生成了 {len(spheres)} 个球，沿 X 轴从 {x_positions[0]} 到 {x_positions[-1]}")

# 分割
all_segments = []
for s in spheres:
    segs = split_sphere_by_box(s, L)
    all_segments.extend(segs)

print(f"分割后总片段数: {len(all_segments)}")
print(f"  预期: {len(spheres)} (因为所有球都在盒内，不应被切割)")

# 连通性
connected, uf = build_connectivity(all_segments, L, thresh)
print(f"导通结果: {connected}")
print(f"  预期: True（从左电极到右电极形成完整通路）")

if connected:
    print("✅ 场景1 通过！")
else:
    print("❌ 场景1 失败！")
    # 调试：检查哪些球没有连通
    left_root = uf.find(("PLANE", "LEFT"))
    right_root = uf.find(("PLANE", "RIGHT"))
    for i, seg in enumerate(all_segments):
        print(f"  片段{i}: 根={uf.find(i)}, 中心={seg.c if hasattr(seg, 'c') else 'N/A'}")
    print(f"  左电极根: {left_root}")
    print(f"  右电极根: {right_root}")


# ============================================================
# 场景2：跨越边界的球（球心在 4900，半径 200）
# 应该被切割成2个片段，且平移后的片段在 -5100
# ============================================================
print("\n\n【场景2】单个跨越边界的球（球心在 4900）")
print("-" * 70)

s_cross = Sphere(np.array([4900, 0.0, 0.0]), r, id=99)
segs_cross = split_sphere_by_box(s_cross, L)

print(f"分割后片段数: {len(segs_cross)}")
print(f"  预期: 2（盒内部分 + 平移回来的部分）")

for i, seg in enumerate(segs_cross):
    if isinstance(seg, Sphere):
        print(f"  片段{i}: Sphere, 中心={seg.c}, 半径={seg.r}")
    elif hasattr(seg, 'c') and hasattr(seg, 'n') and hasattr(seg, 'd'):
        print(f"  片段{i}: SphericalCap, 中心={seg.c}, 法向量={seg.n}, d={seg.d}")
        # 验证几何含义
        if seg.n[0] == -1.0 and abs(seg.d + 100.0) < 1e-6:
            print(f"    ✅ 正确: 表示 x <= 5000")
        elif seg.n[0] == 1.0 and abs(seg.d - 100.0) < 1e-6:
            print(f"    ✅ 正确: 表示 x >= -5000")
        else:
            print(f"    ❌ 法向量或d值异常")


# ============================================================
# 场景3：跨越边界的球 + 辅助球，形成完整通路
# ============================================================
print("\n\n【场景3】跨越边界的球 + 辅助球，形成左右电极通路")
print("-" * 70)

# 球A：跨越右边界，球心在 4900
#   盒内部分在 x=4900，平移部分在 x=-5100
#   平移部分距离左电极（x=-5000）只有 100nm，不接触
#   需要再加一个球桥接平移部分到左电极

# 球B：桥接平移部分到左电极
#   平移部分球心在 -5100，半径 200，左表面在 -5300，右表面在 -4900
#   左电极在 -5000，所以平移部分的右表面在 -4900，距离左电极 100nm
#   需要再加一个球接触左电极，并与平移部分连通

# 球C：接触左电极，球心在 -4800（左表面在 -5000）
#   球C 与平移部分（球心 -5100）的球心距 = 300nm，连通！

# 球D：接触右电极，球心在 4800（右表面在 5000）
#   球D 与球A的盒内部分（球心 4900）的球心距 = 100nm，连通！

spheres2 = [
    Sphere(np.array([4900, 0.0, 0.0]), r, id=0),   # 球A：跨越右边界
    Sphere(np.array([-4900, 0.0, 0.0]), r, id=1),  # 球B：桥接平移部分（-5100）到左电极（-5000）
    Sphere(np.array([-4800, 0.0, 0.0]), r, id=2),  # 球C：接触左电极
    Sphere(np.array([4800, 0.0, 0.0]), r, id=3),   # 球D：接触右电极
]

print("构造的球:")
print(f"  球A: 中心=(4900,0,0), 半径=200 (跨越右边界)")
print(f"  球B: 中心=(-5100,0,0), 半径=200 (桥接平移部分)")
print(f"  球C: 中心=(-4800,0,0), 半径=200 (接触左电极)")
print(f"  球D: 中心=(4800,0,0), 半径=200 (接触右电极)")
print(f"  预期通路: 左电极 ← 球C ← 球B ← 球A平移部分 ← 球A盒内部分 ← 球D → 右电极")

# 分割
all_segments2 = []
for s in spheres2:
    segs = split_sphere_by_box(s, L)
    all_segments2.extend(segs)

print(f"\n分割后总片段数: {len(all_segments2)}")

# 打印每个片段
for i, seg in enumerate(all_segments2):
    if isinstance(seg, Sphere):
        print(f"  片段{i}: Sphere, 中心={seg.c}")
    elif hasattr(seg, 'c') and hasattr(seg, 'n') and hasattr(seg, 'd'):
        print(f"  片段{i}: SphericalCap, 中心={seg.c}, n={seg.n}, d={seg.d}")

# 连通性
connected2, uf2 = build_connectivity(all_segments2, L, thresh)
print(f"\n导通结果: {connected2}")
print(f"  预期: True")

if connected2:
    print("✅ 场景3 通过！")
else:
    print("❌ 场景3 失败！")
    # 详细调试
    left_root = uf2.find(("PLANE", "LEFT"))
    right_root = uf2.find(("PLANE", "RIGHT"))
    print(f"  左电极根: {left_root}")
    print(f"  右电极根: {right_root}")
    for i, seg in enumerate(all_segments2):
        print(f"  片段{i}: 根={uf2.find(i)}, 中心={seg.c if hasattr(seg, 'c') else 'N/A'}")