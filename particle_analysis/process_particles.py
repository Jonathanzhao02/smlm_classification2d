#!/usr/local/Caskroom/miniconda/base/envs/picasso/bin/python3
from time import time
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from clustering import KMeansClusterIdentification, DBSCANClusterIdentification
from align import GridAlignment

VALID_CLUSTER_METHODS = ['kmeans', 'dbscan']

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
GRID_WEIGHTS = np.ones(GRID.shape[0])
GRID_WEIGHTS[ORIENTATION_IDXES] = 1.5

# TEMPLATE 2: 6x8 grid
# TODO:
# Change this to JUST orientation markers
# GRID = np.zeros((48,2), dtype=np.float64)

# for i in range(6):
#     for j in range(8):
#         GRID[i * 8 + j] = [-3.5 + j, -2.5 + i]

# GRID *= 0.1
# ORIENTATION_IDXES = np.array([6, 7, 15, 32, 40, 41, 46, 47, 39])
# GRID_WEIGHTS = np.zeros(GRID.shape[0])
# GRID_WEIGHTS[0] = 1

# KMeans params
CLASS_SWEEP = list(range(3,13)) # for template 1
# CLASS_SWEEP = list(range(9,49)) # for template 2

DISPLAY_INERTIAS = False
KMEANS_THRESHOLD = 0.15 # threshold for inertia
SIZE_THRESHOLD = 0.5 # threshold for cluster size filtering

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing clusters.mat and subParticles.mat")
    parser.add_argument("--cluster", "-c", help="Clustering method to use", default='kmeans')
    args = parser.parse_args()

    f = Path(args.input)
    clusters = loadmat(f.joinpath("clusters.mat"))['clusters'][0]
    subParticles = loadmat(f.joinpath("subParticles.mat"))['subParticles'][0]

    datatype = np.dtype([('points', 'O'), ('sigma', 'O'), ('group', 'S1024'), ('raw_read', 'O'), ('correct', 'O')])

    import code
    code.interact(local=locals())

    cluster_method = args.cluster

    if cluster_method not in VALID_CLUSTER_METHODS:
        raise Exception("Invalid cluster method! Possible values are " + ", ".join(VALID_CLUSTER_METHODS))

    n_particles = 0
    group_avgs = np.zeros(clusters.size)
    group_counts = np.zeros((clusters.size, GRID.shape[0]))
    group_reject_counts = np.zeros(clusters.size)
    global_avg = 0
    raw_reads = np.empty(clusters.size, dtype='O')

    for i in range(clusters.size):
        raw_reads[i] = np.zeros((clusters[i].size, GRID.shape[0]), dtype=bool)

    # Loop over each identified class
    for class_id,groups in enumerate(clusters):
        group_n_particles = groups.size
        n_particles += groups.size

        if group_n_particles == 0:
            continue

        # Loop over each particle in each class
        for group_idx,group in enumerate(groups.flatten()):
            start = time()

            points = subParticles[group - 1]['points'][0][0]
            x = points[:,0]
            y = points[:,1]

            if DISPLAY_PARTICLE:
                plt.figure(figsize=(6,6))
                plt.title(f'Class {class_id} Group {group}')
                plt.plot(x,y,',')
                plt.show()

            if cluster_method == 'kmeans':
                cluster = KMeansClusterIdentification(points)
                print('Optimizing number of KMeans clusters')
                cluster.optimize_clusters(CLASS_SWEEP, KMEANS_THRESHOLD, display_inertia=DISPLAY_INERTIAS)
                cluster.cluster(DISPLAY_FINAL_CLUSTER, SIZE_THRESHOLD)
            elif cluster_method == 'dbscan':
                cluster = DBSCANClusterIdentification(points)
                cluster.cluster(DISPLAY_FINAL_CLUSTER)

            n_clusters = cluster.n_clusters
            centroids = cluster.centroids
            
            # Only for display purposes, can ignore
            if DISPLAY_DISTANCES:
                dist_matrix = np.zeros((n_clusters,n_clusters))

                for i in range(n_clusters - 1):
                    for j in range(i+1, n_clusters):
                        dist_matrix[i,j] = np.linalg.norm(centroids[i] - centroids[j], ord=2)
                
                plt.figure(figsize=(6,6))
                plt.title(f'Class {class_id} Distances')
                plt.plot(x,y,',')

                for i in range(n_clusters):
                    plt.plot(*centroids[i], 'r*')
                
                dist_mean = np.mean(dist_matrix[dist_matrix > 0])
                
                for i in range(n_clusters - 1):
                    for j in range(i+1, n_clusters):
                        if dist_matrix[i,j] < dist_mean:
                            plt.plot([centroids[i][0], centroids[j][0]], [centroids[i][1], centroids[j][1]], 'r--')
                            mid = (centroids[i] + centroids[j]) / 2
                            plt.text(*mid, str(dist_matrix[i,j])[:5], ha='center', va='center')
            
                plt.show()
            
            alignment = GridAlignment(GRID, centroids, GRID_WEIGHTS, 1. / cluster.cluster_sizes)
            print('Calculating rough transform')
            alignment.roughClock(0.18 / 4., 8) # rough rotation with rough translation, args are stepsize and n_steps
            alignment.align([ [-0.18 * 3, 0.18 * 3], [-0.18 * 3, 0.18 * 3], [0.9, 1.1], [0.9, 1.1], [0, 2 * np.pi] ]) # transform bounds
            nn = np.unique(alignment.nn[1]) # read out binary

            # Only for display purposes, can ignore
            if DISPLAY_GRID:
                gridTran = alignment.gridTran
                inv_nn = np.setdiff1d(np.arange(GRID.shape[0]), nn, True)
                plt.figure(figsize=(6,6))
                plt.title(f'Class {class_id} Aligned Template')
                plt.plot(x,y,',')
                plt.plot(centroids[:,0], centroids[:,1], 'r*')
                plt.plot(gridTran[inv_nn,0], gridTran[inv_nn,1], 'k*')
                plt.plot(gridTran[nn,0], gridTran[nn,1], '*', color='#00FF00')
                plt.show()
            
            end = time()

            # Record results
            global_avg += end - start
            group_avgs[class_id] += end - start
            raw_reads[class_id][group_idx][nn] = 1

            # Check orientation markers are present
            if all(np.isin(ORIENTATION_IDXES, nn)):
                group_counts[class_id][nn] += 1
            else:
                group_reject_counts[class_id] += 1
        
        group_avgs[class_id] /= group_n_particles
    
    global_avg /= n_particles

    print("Avg. Times:")
    print(group_avgs)
    print(global_avg)

    # Save all read results
    savemat(str(f.joinpath('raw_reads.mat')), { 'raw_reads': raw_reads })

    # Create visual representation of counts at each template position
    for class_id in range(clusters.size):
        plt.title(f"Class {class_id}")
        counts = group_counts[class_id]

        for i,point in enumerate(GRID):
            plt.text(*point, str(counts[i]), ha='center', va='center')

        plt.show()

    import code
    code.interact(local=locals())
