import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())

from src.geometry import Sphere, Cylinder
from src.truncation import split_sphere_by_box
from src.connectivity import particle_surface_distance, UnionFind
from src.connectivity import build_connectivity

L = 10000.0
r = 200.0
thresh = 1.8

print("="*70)
print("完整诊断：从球到连通性")
print("="*70)

# ============================================================
# 1. 两个球在盒子中间，互相连通但不接触电极
# ============================================================
print("\n【测试1】两个球在盒子中间，球心距400nm（表面距离0）")
print("-"*70)

s1 = Sphere(np.array([0., 0., 0.]), r, id=0)
s2 = Sphere(np.array([400., 0., 0.]), r, id=1)

segs1 = split_sphere_by_box(s1, L)
segs2 = split_sphere_by_box(s2, L)
all_segs = segs1 + segs2

print(f"分割后片段数: {len(all_segs)}")

# 手动检查距离
dist_manual = particle_surface_distance(all_segs[0], all_segs[1])
print(f"particle_surface_distance: {dist_manual}")

# 手动并查集
uf_manual = UnionFind()
for i in range(len(all_segs)):
    for j in range(i+1, len(all_segs)):
        d = particle_surface_distance(all_segs[i], all_segs[j])
        if d <= thresh:
            uf_manual.union(i, j)
print(f"手动并查集: 片段0根={uf_manual.find(0)}, 片段1根={uf_manual.find(1)}")
print(f"手动并查集连通: {uf_manual.find(0) == uf_manual.find(1)}")

# build_connectivity
connected, uf_built = build_connectivity(all_segs, L, thresh)
print(f"build_connectivity 返回: {connected}")
print(f"  LEFT_NODE根: {uf_built.find(('PLANE', 'LEFT'))}")
print(f"  RIGHT_NODE根: {uf_built.find(('PLANE', 'RIGHT'))}")
print(f"  片段0根: {uf_built.find(0)}")
print(f"  片段1根: {uf_built.find(1)}")

# ============================================================
# 2. 检查 build_connectivity 内部是否执行了 union
# ============================================================
print("\n【测试2】在 build_connectivity 调用后，检查片段是否与 LEFT_NODE 连接")
print("-"*70)

# 检查片段0是否连接到了 LEFT_NODE
left_root = uf_built.find(('PLANE', 'LEFT'))
print(f"LEFT_NODE 根: {left_root}")
for i, seg in enumerate(all_segs):
    root_i = uf_built.find(i)
    print(f"  片段{i} 根: {root_i}")
    if root_i == left_root:
        print(f"    -> 片段{i} 连接到左电极！")

print("\n" + "="*70)
print("【测试3】球接触左电极")
print("="*70)

# 球1接触左电极（球心-4800，半径200，左表面-5000）
s_left = Sphere(np.array([-4800., 0., 0.]), r, id=0)
segs_left = split_sphere_by_box(s_left, L)

connected_left, uf_left = build_connectivity(segs_left, L, thresh)
print(f"单个球接触左电极: {connected_left}")
print(f"预期: False（因为没有右电极）")

# 检查这个球是否连接到了 LEFT_NODE
left_root = uf_left.find(('PLANE', 'LEFT'))
print(f"LEFT_NODE 根: {left_root}")
for i, seg in enumerate(segs_left):
    print(f"  片段{i} 根: {uf_left.find(i)}")
    if uf_left.find(i) == left_root:
        print(f"    ✅ 片段{i} 连接到左电极！")

print("\n" + "="*70)
print("【测试4】两个球：一个接触左电极，另一个接触右电极，中间连通")
print("="*70)

# 球1接触左电极
s1 = Sphere(np.array([-4800., 0., 0.]), r, id=0)
# 球2在中间，与球1连通（球心距400）
s2 = Sphere(np.array([-4400., 0., 0.]), r, id=1)
# 球3在中间，与球2连通
s3 = Sphere(np.array([-4000., 0., 0.]), r, id=2)
# ... 一直到球N接触右电极
# 用33个球覆盖 -4800 到 4800
x_pos = np.linspace(-4800, 4800, 33)
spheres = [Sphere(np.array([x, 0., 0.]), r, id=i) for i, x in enumerate(x_pos)]

segments = []
for s in spheres:
    segments.extend(split_sphere_by_box(s, L))

connected_chain, uf_chain = build_connectivity(segments, L, thresh)
print(f"33个球从左到右: {connected_chain}")
print(f"预期: True")

if connected_chain:
    print("✅ 完整通路形成！")
else:
    print("❌ 完整通路未形成")
    # 检查左右电极各连接了多少片段
    left_root = uf_chain.find(('PLANE', 'LEFT'))
    right_root = uf_chain.find(('PLANE', 'RIGHT'))
    left_count = sum(1 for i in range(len(segments)) if uf_chain.find(i) == left_root)
    right_count = sum(1 for i in range(len(segments)) if uf_chain.find(i) == right_root)
    print(f"连接左电极的片段数: {left_count}")
    print(f"连接右电极的片段数: {right_count}")


print("\n" + "="*70)
print("验证球冠-圆柱距离")
print("="*70)

from src.geometry import Cylinder, Sphere, SphericalCap
from src.geometry import sphere_cylinder_surface_distance
from src.geometry import sphericalcap_cylinder_surface_distance

# 创建一个圆柱
cyl = Cylinder(np.array([0., -100., 0.]), np.array([0., 100., 0.]), 30.0, id=0)

# 创建一个球冠（球心在4900，在x=5000处切割）
cap = SphericalCap(np.array([4900., 0., 0.]), 200.0, np.array([-1., 0., 0.]), -100.0, id=1)

# 完整球-圆柱距离（作为参考）
dummy_sphere = Sphere(cap.c, cap.r, id=cap.id)
dist_full = sphere_cylinder_surface_distance(dummy_sphere, cyl)
print(f"完整球-圆柱距离: {dist_full}")

# 球冠-圆柱距离
dist_cap = sphericalcap_cylinder_surface_distance(cap, cyl)
print(f"球冠-圆柱距离: {dist_cap}")

# 球冠在x=4900，圆柱在x=0附近，距离应该约4700nm
print(f"预期距离: 约4700nm")