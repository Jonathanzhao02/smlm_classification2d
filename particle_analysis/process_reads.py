#!/usr/local/Caskroom/miniconda/base/envs/picasso/bin/python3
from time import time
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# General params
DISPLAY_PARTICLE = False
DISPLAY_FINAL_CLUSTER = False
DISPLAY_DISTANCES = False
DISPLAY_GRID = False

# Template + Weights

# TEMPLATE 1: 3x3 grid w/ 3 offset orientation markers
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
], dtype=np.float64) * 0.18

ORIENTATION_IDXES = np.array([0, 1, 2])
INV_ORIENTATION_IDXES = np.setdiff1d(np.arange(GRID.shape[0]), ORIENTATION_IDXES)
GRID_WEIGHTS = np.ones(GRID.shape[0])
GRID_WEIGHTS[ORIENTATION_IDXES] = 1.5

# Used to convert read binary into actual index / letter pairing
LETTER_VALUES = np.array(
    [0, 0, 0, 32, 16, 8, 4, 2, 1, 0, 0, 0]
)

IDX_VALUES = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1]
)

def to_string(val):
    idx = val & 0b111
    letter = val >> 3
    return f"{letter}/{chr(letter + ord('A') - 1)},{idx}"

# TEMPLATE 2: 6x8 grid
# TODO:
# Change this to JUST orientation markers
# GRID = np.zeros((48,2), dtype=np.float64)

# for i in range(6):
#     for j in range(8):
#         GRID[i * 8 + j] = [-3.5 + j, -2.5 + i]

# GRID *= 0.1
# ORIENTATION_IDXES = np.array([6, 7, 15, 32, 40, 41, 46, 47, 39])
# INV_ORIENTATION_IDXES = np.setdiff1d(np.arange(GRID.shape[0]), ORIENTATION_IDXES)
# GRID_WEIGHTS = np.zeros(GRID.shape[0])
# GRID_WEIGHTS[0] = 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing raw_reads.mat, clusters.mat, and subParticles.mat")
    args = parser.parse_args()

    f = Path(args.input)
    read_data = loadmat(f.joinpath("raw_reads.mat"))
    raw_reads = read_data['raw_reads'][0]

    clusters = loadmat(f.joinpath("clusters.mat"))['clusters'][0]
    subParticles = loadmat(f.joinpath("subParticles.mat"))['subParticles'][0]

    datatype = np.dtype([('points', 'O'), ('sigma', 'O'), ('group', 'S1024'), ('raw_read', 'O'), ('correct', 'O')])

    group_counts = np.zeros((raw_reads.size, GRID.shape[0]))
    group_reject_counts = np.zeros(raw_reads.size)
    readouts = np.empty(raw_reads.size, dtype='O')

    # Loop over each identified class
    for class_id,reads in enumerate(raw_reads):
        readouts[class_id] = np.zeros(reads.shape[0], dtype=int)

        # Loop over each particle in each class
        for particle_idx,read in enumerate(reads):
            # Check orientation markers are present
            if np.sum(read[ORIENTATION_IDXES]) == ORIENTATION_IDXES.size:
                group_counts[class_id] += read
            else:
                group_reject_counts[class_id] += 1
            
            readouts[class_id][particle_idx] = (np.sum(read * LETTER_VALUES) << 3) + np.sum(read * IDX_VALUES)
        
        plt.title(f"Class {class_id}")
        [n, bins, _] = plt.hist(readouts[class_id], bins=range(1 << INV_ORIENTATION_IDXES.size))
        plt.ylim(top=reads.shape[0])
        plt.show()

        top_n = bins[np.argsort(n)[::-1][:readouts.shape[0]]].astype(int)
        print(np.sort(n)[::-1][:readouts.shape[0]])
        print(list(map(to_string, top_n)))
    
    all_readouts = np.concatenate(readouts)
    plt.title("Overall")
    [n, bins, _] = plt.hist(all_readouts, bins=range(1 << INV_ORIENTATION_IDXES.size))
    plt.ylim(top=all_readouts.shape[0])
    plt.show()

    top_n = bins[np.argsort(n)[::-1][:readouts.shape[0]]].astype(int)
    print(np.sort(n)[::-1][:readouts.shape[0]])
    print(list(map(to_string, top_n)))

    # Create visual representation of counts at each template position
    for class_id in range(raw_reads.size):
        plt.title(f"Class {class_id}")
        counts = group_counts[class_id]

        for i,point in enumerate(GRID):
            plt.text(*point, str(counts[i]), ha='center', va='center')

        plt.show()

# 0. make poster (come up with outline for next week)
# 2. identify misclassifications
# 4. fix MATLAB 1 group error (combine with nearby clusters)
# 5. scale sweep + increase nAngles 2x
# 6. elbow on KMeans on MDS
# 7. start on QR deformation correction
#    a. 2/3 redundancy, 10 data points for each class
# 7.5. do conventional image analysis, preprocess for all others
# 8. fine tune model architecture
# 9. conventional (MLE) comparison

# crytographic hash function
