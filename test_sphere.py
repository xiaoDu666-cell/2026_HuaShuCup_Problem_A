import numpy as np
from src.geometry import Sphere
from src.truncation import split_sphere_by_box

L = 10000.0
r = 200.0

# 球心在4900，半径200，越过 x=5000 边界
s = Sphere(np.array([4900, 0, 0]), r, id=0)
segs = split_sphere_by_box(s, L)

print(f"分割后 {len(segs)} 个片段:")
for seg in segs:
    print(f"  类型: {type(seg).__name__}, 中心: {seg.c}, d={seg.d if hasattr(seg, 'd') else 'N/A'}")

for seg in segs:
    print(f"  类型: {type(seg).__name__}, 中心: {seg.c}, n={seg.n}, d={seg.d}")