# 在 src/__init__.py 中添加
from .geometry import Cylinder, Sphere, SphericalCap
from .truncation import split_cylinder_by_box, split_sphere_by_box
from .connectivity import build_connectivity, particle_surface_distance