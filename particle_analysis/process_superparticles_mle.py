from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from align import LocalizationCluster

# General params
DISPLAY_PARTICLE = True
DISPLAY_FINAL_CLUSTER = False
DISPLAY_DISTANCES = False
DISPLAY_GRID = True

# Template + Weights

# TEMPLATE 1: 3x3 grid w/ 3 offset orientation markers
SCALE = 0.18
GRID = np.array([
    [1.5, 1.5],
    [1.5, 2.5],
    [0.5, 2.5],
    [-1, 1],
    [0, 1],
    [1, 1],
    [-1, 0],
    [0, 0],
    [1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
], dtype=np.float64) * SCALE

ORIENTATION_IDXES = np.array([0, 1, 2])
INV_ORIENTATION_IDXES = np.setdiff1d(np.arange(GRID.shape[0]), ORIENTATION_IDXES)
GRID_WEIGHTS = np.ones(GRID.shape[0])
GRID_WEIGHTS[ORIENTATION_IDXES] = 1.5
BOUNDS = [[-SCALE * 3, SCALE * 3], [-SCALE * 3, SCALE * 3], [0.9, 1.1], [0.9, 1.1], [0, 2 * np.pi]]

# TEMPLATE 2: 6x8 grid
# SCALE = 0.08
# GRID = np.zeros((48,2), dtype=np.float64)

# # 0  1  2  3  4  5  6  7
# # 8  9  10 11 12 13 14 15
# # 16 17 18 19 20 21 22 23
# # 24 25 26 27 28 29 30 31
# # 32 33 34 35 36 37 38 39
# # 40 41 42 43 44 45 46 47
# # Keep in mind origami is FLIPPED VERTICALLY from image -> matplotlib

# for i in range(6):
#     for j in range(8):
#         GRID[i * 8 + j] = [-3.5 + j, 2.5 - i]

# GRID *= SCALE
# ORIENTATION_IDXES = np.array([0, 1, 8, 6, 7, 15, 32, 40, 41])
# INV_ORIENTATION_IDXES = np.setdiff1d(np.arange(GRID.shape[0]), ORIENTATION_IDXES)
# GRID_WEIGHTS = np.ones(GRID.shape[0])
# GRID_WEIGHTS[ORIENTATION_IDXES] = 0.2
# BOUNDS = [[-SCALE * 3, SCALE * 3], [-SCALE * 3, SCALE * 3], [0.9, 1.1], [0.9, 1.1], [0, 2 * np.pi]]

# # 3-repetition
# # REPETITION_PAIRS = np.array([
# #     [14, 22, 23], [5, 13, 21], [4, 12, 20], [3, 11, 19], [2, 10, 18], [9, 16, 17],
# #     [30, 31, 38], [29, 37, 45], [28, 36, 44], [27, 35, 43], [26, 34, 42], [24, 25, 33],
# # ], dtype=int)

# # 2-repetition
# REPETITION_PAIRS = np.array([
#     [14, 22], [13, 21], [12, 20], [11, 19], [10, 18], [9, 17],
#     [30, 38], [29, 37], [28, 36], [27, 35], [26, 34], [25, 33],
# ], dtype=int)

# def apply_repetition(raw):
#     read = np.zeros(REPETITION_PAIRS.shape[0], dtype=int)

#     for i,pair in enumerate(REPETITION_PAIRS):
#         read[i] = any(raw[pair])
    
#     return read

# def true_read(tag):
#     c = tag[0]

#     # 1, 1
#     if c == 'A':
#         return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
#     # 19, 2
#     elif c == 'S':
#         return [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
#     # 21, 3
#     elif c == 'U':
#         return [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1]

# def read_match(read, tag):
#     read = apply_repetition(read)
#     return all(read == true_read(tag))

# calculated through average x/y uncertainty across all localizations
# BANDWIDTH = 0.06 # empirical for NSF
BANDWIDTH = 0.0435 # for NSF
# BANDWIDTH = 0.0238 # for 3-repetition ASU
# BANDWIDTH = 0.035 # empirical for 2-repetition ASU
# BANDWIDTH = 0.0323 # for 2-repetition ASU

# For template 1
TOP_N_CLUSTERS = 3
SIZE_THRESHOLD_MEANSHIFT = 0.3

# For template 2
# TOP_N_CLUSTERS = 9
# SIZE_THRESHOLD_MEANSHIFT = 0.6

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing classes.mat")
    args = parser.parse_args()

    f = Path(args.input).joinpath("classes.mat")
    classes = loadmat(f)['classes'][0]

    # Loop over each identified superparticle class
    for class_id in range(len(classes)):
        points = classes[class_id][0,-1]
        x = points[:,0]
        y = points[:,1]
        sigma = np.zeros(x.size)
        sigma[:] = BANDWIDTH

        if DISPLAY_PARTICLE:
            plt.figure(figsize=(6,6))
            plt.title(f'Class {class_id}')
            plt.plot(x,y,',')
            plt.show()

        alignment = LocalizationCluster(GRID, np.array([x, y, sigma]).T, BANDWIDTH, False)
        print('Calculating transform')
        cost, _ = alignment.fitAndPixelate((GRID.shape[0],1), SCALE, 5000, 1e-6)
        hist = alignment.hist[:-1]
        mean_size = np.mean(np.sort(hist)[::-1][:TOP_N_CLUSTERS])
        size_thresh = mean_size * SIZE_THRESHOLD_MEANSHIFT
        hist[hist < size_thresh] = 0
        hist[hist >= size_thresh] = 1

        if DISPLAY_GRID:
            gridTran = alignment.gridTran
            plt.figure(figsize=(6,6))
            plt.title(f'Class {class_id} Aligned Template')
            plt.scatter(x,y,s=np.ones(x.size),alpha=0.3)
            plt.scatter(gridTran[:,0], gridTran[:,1], c=hist)
            plt.show()
