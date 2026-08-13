# truncation.py
# Boundary truncation with periodic wrapping:
# Split cylinder axis by intersections with box faces (including periodic images),
# then map each resulting piece back into the primary box [-L/2, L/2]^3 by
# translating by integer multiples of L so that the piece's midpoint lies in the box.
from __future__ import annotations
import numpy as np
from typing import List
from .geometry import Cylinder

def _map_point_into_box(pt: np.ndarray, L: float) -> np.ndarray:
    """Map a point into the primary box [-L/2, L/2]^3 using periodic wrapping."""
    half = L / 2.0
    # map each coordinate to [-half, half)
    # use modulo that handles negative values correctly
    mapped = ((pt + half) % L) - half
    # Handle the edge case when mapped == -half and original > half-eps, clamp to half
    # but for our purposes the above is sufficient.
    return mapped

def split_cylinder_by_box(cyl: Cylinder, L: float) -> List[Cylinder]:
    """
    Split a cylinder axis into segments at box faces (including faces of periodic images),
    then map each resulting segment back into the primary box by translating by integer
    multiples of L so the segment midpoint falls inside the box.

    This implements the periodic boundary 'edge truncation' rule: parts of the
    cylinder that lie outside the box are shifted back from the exceeded side
    into the box by +/- L (or multiples thereof).

    Returns a list of Cylinder segments (each with same radius and same original id).
    """
    A = np.asarray(cyl.p0, dtype=float)
    B = np.asarray(cyl.p1, dtype=float)
    d = B - A
    half = L / 2.0
    SMALL = 1e-15

    # Collect parameter values t where the axis intersects box faces (including periodic images)
    ts = [0.0, 1.0]

    # For each axis, find face positions that lie between the min and max coordinate of the segment
    for dim in range(3):
        a_coord = A[dim]
        b_coord = B[dim]
        min_coord = min(a_coord, b_coord)
        max_coord = max(a_coord, b_coord)
        # Faces in the infinite grid occur at positions: -half + n*L and +half + n*L (redundant)
        # We'll generate faces at -half + n*L and +half + n*L for n in a range covering [min_coord, max_coord]
        # Compute n range roughly:
        # find n such that face_position in [min_coord, max_coord]
        # For -half + n*L:
        if abs(d[dim]) < SMALL:
            # axis parallel to this plane direction: no t from this axis (unless the entire segment is outside,
            # which will be handled by midpoint mapping below)
            continue

        # determine reasonable n range
        # solve -half + n*L in [min_coord, max_coord] -> n in [(min_coord + half)/L, (max_coord + half)/L]
        n_min = int(np.floor((min_coord + half) / L)) - 1
        n_max = int(np.ceil((max_coord + half) / L)) + 1

        # also include +half + n*L faces (they are offset by L/2 but will be covered by above set in a shifted n).
        # We'll explicitly include both sets to be robust.
        face_positions = []
        for n in range(n_min, n_max + 1):
            face_positions.append(-half + n * L)
            face_positions.append(half + n * L)

        # compute t for each face
        for face in face_positions:
            denom = (B[dim] - A[dim])
            if abs(denom) < SMALL:
                continue
            t = (face - A[dim]) / denom
            if 0.0 < t < 1.0:
                ts.append(float(t))

    # unique & sort
    ts = sorted(set(ts))

    segments: List[Cylinder] = []
    for i in range(len(ts) - 1):
        t0 = ts[i]
        t1 = ts[i + 1]
        # skip degenerate intervals
        if t1 - t0 <= 1e-15:
            continue
        p0 = A + d * t0
        p1 = A + d * t1
        midpoint = 0.5 * (p0 + p1)

        # Map midpoint into primary box, compute shift vector to move entire segment so that midpoint is inside box
        mapped_mid = _map_point_into_box(midpoint, L)
        shift = mapped_mid - midpoint

        p0_shift = p0 + shift
        p1_shift = p1 + shift

        # Numeric clamp to box boundaries to avoid tiny out-of-bound values
        p0_shift = np.minimum(np.maximum(p0_shift, -half), half)
        p1_shift = np.minimum(np.maximum(p1_shift, -half), half)

        # Append segment as Cylinder within primary box
        segments.append(Cylinder(p0_shift, p1_shift, cyl.r, id=cyl.id))

    return segments