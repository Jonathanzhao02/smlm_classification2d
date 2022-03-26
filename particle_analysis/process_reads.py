from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

# General params
DISPLAY_HISTOGRAM = False
DISPLAY_TEMPLATES = False

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

def true_read(tag):
    c = tag[0]

    # 14, 0
    if c == 'N':
        return [1,1,1,0, 0, 1, 1, 1, 0, 0, 0, 0]
    # 19, 1
    elif c == 'S':
        return [1,1,1,0, 1, 0, 0, 1, 1, 0, 0, 1]
    # 6, 2
    elif c == 'F':
        return [1,1,1,0, 0, 0, 1, 1, 0, 0, 1, 0]

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
    parser.add_argument("input", help="Path to folder containing final.mat")
    parser.add_argument("--out", "-o", help="Name of output folder", default="")
    args = parser.parse_args()

    f = Path(args.input)
    out_f = f.joinpath(args.out or 'final')
    out_f.mkdir(exist_ok=True)

    picks = loadmat(f.joinpath("final.mat"))['picks'][0]
    n_picks = picks.size

    def reduce_ar(a):
        while hasattr(a, 'shape') and len(a.shape) and a.shape[0] == 1:
            a = a[0]
        return a

    extract_field = lambda s: np.vectorize(lambda x: reduce_ar(x[s]), otypes='O')(picks)

    raw_reads = np.stack(extract_field('raw_read')).astype(int)
    correct_reads = extract_field('correct').astype(bool)
    groups = extract_field('group').astype('S1024')
    points = extract_field('points')
    centroids = extract_field('centroids')
    grids = extract_field('grid')
    costs = extract_field('cost')
    clusters = extract_field('cluster').astype(int)
    n_clusters = np.unique(clusters).size

    clusters_sizes = np.zeros(n_clusters, dtype=int)

    for i in range(n_clusters):
        clusters_sizes[i] = np.sum(clusters == i)

    group_counts = np.zeros((n_clusters, GRID.shape[0]), dtype=int)
    group_reject_counts = np.zeros(n_clusters, dtype=int)
    readouts = np.empty(n_clusters, dtype='O')

    for i in range(n_clusters):
        readouts[i] = np.zeros(clusters_sizes[i], dtype=int)
    
    inds = np.zeros(n_clusters, dtype=int)

    for i in range(n_picks):
        read = raw_reads[i]
        cluster = clusters[i]

        # Check orientation markers are present
        if np.sum(read[ORIENTATION_IDXES]) == ORIENTATION_IDXES.size:
            group_counts[cluster] += read
        else:
            group_reject_counts[cluster] += 1
        
        readouts[cluster][inds[cluster]] = (np.sum(read * LETTER_VALUES) << 3) + np.sum(read * IDX_VALUES)
        inds[cluster] += 1
    
    if DISPLAY_HISTOGRAM:
        for i in range(n_clusters):
            plt.title(f"Class {i}")
            [n, bins, _] = plt.hist(readouts[i], bins=range(1 << INV_ORIENTATION_IDXES.size))
            plt.ylim(top=clusters_sizes[i])
            plt.show()

            top_n = bins[np.argsort(n)[::-1][:n_clusters]].astype(int)
            print(np.sort(n)[::-1][:n_clusters])
            print(list(map(to_string, top_n)))
        
        all_readouts = np.concatenate(readouts)
        plt.title("Overall")
        [n, bins, _] = plt.hist(all_readouts, bins=range(1 << INV_ORIENTATION_IDXES.size))
        plt.ylim(top=n_picks)
        plt.show()

        top_n = bins[np.argsort(n)[::-1][:n_clusters]].astype(int)
        print(np.sort(n)[::-1][:n_clusters])
        print(list(map(to_string, top_n)))

    if DISPLAY_TEMPLATES:
        # Create visual representation of counts at each template position
        for class_id in range(n_clusters):
            plt.title(f"Class {class_id}")
            counts = group_counts[class_id]

            for i,point in enumerate(GRID):
                plt.text(*point, str(counts[i]), ha='center', va='center')

            x = GRID[:,0]
            y = GRID[:,1]

            plt.xlim(x.min() * 1.5,x.max() * 1.5)
            plt.ylim(y.min() * 1.5,y.max() * 1.5)

            plt.show()
    
    print(f"Correctly read {correct_reads.sum()} / {n_picks} picks, {correct_reads.sum() / n_picks * 100:.2f}%")

    out_path = out_f.joinpath('misclassifications')
    out_path.mkdir(exist_ok=True)
    incorrect_reads = np.logical_not(correct_reads)
    raw_reads_mis = raw_reads[incorrect_reads]
    groups_mis = groups[incorrect_reads]
    points_mis = points[incorrect_reads]
    centroids_mis = centroids[incorrect_reads]
    costs_mis = costs[incorrect_reads]
    clusters_mis = clusters[incorrect_reads]
    grids_mis = grids[incorrect_reads]
    n_picks_mis = incorrect_reads.sum()

    for i in range(n_picks_mis):
        pick_group = bytes.decode(groups_mis[i])
        pick_target = true_read(pick_group)
        pick_points = points_mis[i]
        pick_centroids = centroids_mis[i]
        pick_grid = grids_mis[i]
        pick_read = raw_reads_mis[i].astype(bool)
        pick_cost = costs_mis[i]

        pick_val = (np.sum(pick_read * LETTER_VALUES) << 3) + np.sum(pick_read * IDX_VALUES)

        inv_read = np.logical_not(pick_read)
        plt.figure(figsize=(6,6))
        plt.title(f'Pick {pick_group} Aligned Template, Read {to_string(pick_val)}, Cost {pick_cost:.3e}')
        plt.plot(pick_points[:,0],pick_points[:,1],',')
        plt.plot(pick_centroids[:,0], pick_centroids[:,1], 'r*')
        plt.plot(pick_grid[inv_read,0], pick_grid[inv_read,1], 'k*')
        plt.plot(pick_grid[pick_read,0], pick_grid[pick_read,1], '*', color='#00FF00')
        plt.savefig(out_path.joinpath(f"{pick_group}.png"))
        plt.close()


# 0. make poster (come up with outline for next week)
# 1?. add random starting point
# 2?. eccentricity (idea below)
# 4. fix MATLAB 1 group error (combine with nearby clusters)
# 5. scale sweep + increase nAngles 2x
# 6. elbow on KMeans on MDS
# 6.5. rerun everything on repetition code data
# 7. start on QR deformation correction
#    a. 2/3 redundancy, 10 data points for each class
# 7.5. do conventional image analysis, preprocess for all others
# 8. fine tune model architecture
# 9. conventional (MLE) comparison

# idea: measure eccentricity of cluster set of points + size/radius, filter number of groups using that

# crytographic hash function
