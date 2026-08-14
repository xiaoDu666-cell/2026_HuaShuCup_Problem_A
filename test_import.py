print("Step 1: Starting...")

print("Step 2: Importing os, sys...")
import os
import sys
print("Step 3: Done")

print("Step 4: Importing numpy...")
import numpy as np
print("Step 5: Done")

print("Step 6: Importing pandas...")
import pandas as pd
print("Step 7: Done")

print("Step 8: Importing multiprocessing...")
import multiprocessing
print("Step 9: Done")

print("Step 10: Importing from src.geometry...")
from src.geometry import Cylinder, Sphere
print("Step 11: Done")

print("Step 12: Importing from src.monte_carlo...")
from src.monte_carlo import run_single_trial_mixed
print("Step 13: Done")

print("Step 14: Importing from src.truncation...")
from src.truncation import split_cylinder_by_box, split_sphere_by_box
print("Step 15: Done")

print("Step 16: Importing from src.connectivity...")
from src.connectivity import build_connectivity
print("Step 17: All imports successful!")