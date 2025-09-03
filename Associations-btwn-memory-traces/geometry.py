# geometry.py
"""
3D grid placement and pairwise Euclidean distances (Section 2.1).
All 540 network neurons (432 E + 108 I) are placed on a 6×6×15 grid.

Used for distance-dependent Bernoulli rules p(d)=c·exp(-d/λ) (eq. 7).
"""
import numpy as np
from config import CFG

def grid_positions():
    nx, ny, nz = CFG.grid_shape
    coords = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'), axis=-1).reshape(-1,3)
    assert coords.shape[0] == CFG.N_E + CFG.N_I
    return coords.astype(np.float32)

def pairwise_dist(coords):
    # distances in "grid units" as in eq. (7)
    diffs = coords[:,None,:] - coords[None,:,:]
    return np.linalg.norm(diffs, axis=-1).astype(np.float32)
