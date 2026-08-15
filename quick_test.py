import numpy as np
from src.geometry import Sphere
from src.truncation import split_sphere_by_box
from src.connectivity import build_connectivity

# 手动实现 sample_random_spheres（与你的脚本一致）
def sample_random_spheres(N, Lbox, rad, seed=None):
    if N <= 0:
        return []
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-Lbox/2, Lbox/2, size=(N, 3))
    spheres = []
    for i in range(N):
        spheres.append(Sphere(centers[i], rad, id=i))
    return spheres

L = 10000.0
r = 200.0
thresh = 1.8
N = 299
seed = 12345

# 生成球
spheres = sample_random_spheres(N, L, r, seed=seed)
print(f"生成了 {len(spheres)} 个球")

# 切割
segments = []
cut_count = 0
for s in spheres:
    segs = split_sphere_by_box(s, L)
    if len(segs) > 1:
        cut_count += 1
    segments.extend(segs)
print(f"被切割的球数: {cut_count}")
print(f"总片段数: {len(segments)}")

# 打印切割后的片段类型
types = {}
for seg in segments:
    t = type(seg).__name__
    types[t] = types.get(t, 0) + 1
print(f"片段类型分布: {types}")

# 连通性
connected, uf = build_connectivity(segments, L, thresh)
print(f"导通: {connected}")

if not connected:
    LEFT_NODE = ("PLANE", "LEFT")
    RIGHT_NODE = ("PLANE", "RIGHT")
    left_root = uf.find(LEFT_NODE)
    right_root = uf.find(RIGHT_NODE)
    
    left_count = 0
    right_count = 0
    for i, seg in enumerate(segments):
        if uf.find(i) == left_root:
            left_count += 1
        if uf.find(i) == right_root:
            right_count += 1
    print(f"连接到左电极的片段数: {left_count}")
    print(f"连接到右电极的片段数: {right_count}")
    
    print("\n前10个片段:")
    for i in range(min(10, len(segments))):
        seg = segments[i]
        print(f"  片段{i}: 类型={type(seg).__name__}, 根={uf.find(i)}")
        if hasattr(seg, 'c'):
            print(f"    中心={seg.c}")
        if hasattr(seg, 'n') and hasattr(seg, 'd'):
            print(f"    n={seg.n}, d={seg.d}")

# 检查球心是否在边界附近
print("\n球心在边界附近的球 (|x|>4800):")
for s in spheres[:20]:  # 只检查前20个
    if abs(s.c[0]) > 4800 or abs(s.c[1]) > 4800 or abs(s.c[2]) > 4800:
        print(f"  球中心={s.c}")

# 在 quick_test.py 末尾添加

print("\n" + "="*60)
print("最简确定性测试：3个球从左电极到右电极")
print("="*60)

# 球1：接触左电极（左表面在 x=-5000）
# 球心在 -4800，半径200 → 左表面在 -5000
s1 = Sphere(np.array([-4800.0, 0.0, 0.0]), r, id=0)

# 球2：在中间，与球1和球3都连通
# 球心在 -4400，球1球心距400 → 表面距离0，连通
s2 = Sphere(np.array([-4400.0, 0.0, 0.0]), r, id=1)

# 球3：接触右电极（右表面在 x=5000）
# 球心在 4800，半径200 → 右表面在 5000
s3 = Sphere(np.array([4800.0, 0.0, 0.0]), r, id=2)

# 但球1到球3距离太远（-4400 到 4800，需要更多球）
# 改成连续排列的球，间距300nm
print("\n连续排列33个球，间距300nm，从左电极到右电极")
x_positions = np.arange(-4800, 4801, 300)  # -4800, -4500, ..., 4800
spheres_chain = []
for i, x in enumerate(x_positions):
    spheres_chain.append(Sphere(np.array([x, 0.0, 0.0]), r, id=i))

segments_chain = []
for s in spheres_chain:
    segs = split_sphere_by_box(s, L)
    segments_chain.extend(segs)

print(f"球数: {len(spheres_chain)}, 片段数: {len(segments_chain)}")
connected_chain, uf_chain = build_connectivity(segments_chain, L, thresh)
print(f"导通: {connected_chain}")

if not connected_chain:
    print("⚠️ 确定性测试失败！说明 build_connectivity 有严重问题")
    # 打印电极连接情况
    LEFT_NODE = ("PLANE", "LEFT")
    RIGHT_NODE = ("PLANE", "RIGHT")
    left_root = uf_chain.find(LEFT_NODE)
    right_root = uf_chain.find(RIGHT_NODE)
    print(f"左电极根: {left_root}")
    print(f"右电极根: {right_root}")
    for i in range(min(5, len(segments_chain))):
        seg = segments_chain[i]
        print(f"  片段{i}: 根={uf_chain.find(i)}, 中心={seg.c}")
else:
    print("✅ 确定性测试通过！build_connectivity 正常")