import numpy as np
from src.geometry import Sphere
from src.truncation import split_sphere_by_box
from src.connectivity import build_connectivity

L = 10000.0
r = 200.0
thresh = 1.8

print("=" * 60)
print("测试1: 两个完全在盒内的球，互相连通，但不接触电极")
print("-" * 60)

s1 = Sphere(np.array([0, 0, 0]), r, id=0)
s2 = Sphere(np.array([400, 0, 0]), r, id=1)  # 球心距400nm，表面距离0

segs1 = split_sphere_by_box(s1, L)
segs2 = split_sphere_by_box(s2, L)
all_segs = segs1 + segs2

print(f"片段数: {len(all_segs)}")
connected, uf = build_connectivity(all_segs, L, thresh)
print(f"连通: {connected} (预期: False，因为没有球接触电极)")
print()

print("=" * 60)
print("测试2: 两个球接触左电极")
print("-" * 60)

# 球1在左电极附近，表面刚好接触电极
s3 = Sphere(np.array([-4800, 0, 0]), r, id=2)  # 球心在-4800，半径200，表面在-5000
segs3 = split_sphere_by_box(s3, L)

# 球2在球1右侧，与球1连通
s4 = Sphere(np.array([-4400, 0, 0]), r, id=3)  # 球心距400nm
segs4 = split_sphere_by_box(s4, L)

all_segs2 = segs3 + segs4
print(f"片段数: {len(all_segs2)}")
connected2, uf2 = build_connectivity(all_segs2, L, thresh)
print(f"连通: {connected2} (预期: True，因为球1接触左电极，球1-球2连通)")
print()

print("=" * 60)
print("测试3: 跨越边界的球在电极附近")
print("-" * 60)

# 球心在4900，半径200，跨越右边界
# 边界截断后：盒内部分在x=4900，平移部分在x=-5100
s5 = Sphere(np.array([4900, 0, 0]), r, id=4)
segs5 = split_sphere_by_box(s5, L)
print(f"跨越边界球分割后片段数: {len(segs5)}")
for seg in segs5:
    print(f"  {type(seg).__name__}: 中心={seg.c}")

# 检查平移后的部分（在x=-5100）是否接触左电极（在x=-5000）
# 球心在-5100，半径200，表面在-4900，距离左电极100nm，不接触
# 所以单独一个跨越边界的球不应该导通
connected3, uf3 = build_connectivity(segs5, L, thresh)
print(f"单独跨越边界球是否导通: {connected3} (预期: False)")

print()

print("=" * 60)
print("测试4: 跨越边界的球 + 另一个球连接到右电极")
print("-" * 60)

# 球5的盒内部分在x=4900，表面在x=5100（但被截断，实际在x=5000边界）
# 球6在x=4800，半径200，表面在x=5000，与球5的盒内部分连通
s6 = Sphere(np.array([4600, 0, 0]), r, id=5)  # 球心在4600，与球5球心距300nm，连通
segs6 = split_sphere_by_box(s6, L)

# 球6接触右电极？球心4600+200=4800 < 5000，不接触
# 需要再加一个球接触右电极
s7 = Sphere(np.array([4800, 0, 0]), r, id=6)  # 球心4800+200=5000，刚好接触右电极
segs7 = split_sphere_by_box(s7, L)

all_segs4 = segs5 + segs6 + segs7
print(f"总片段数: {len(all_segs4)}")
connected4, uf4 = build_connectivity(all_segs4, L, thresh)
print(f"连通: {connected4} (预期: True，形成左电极←平移部分←盒内部分←球6←球7→右电极)")