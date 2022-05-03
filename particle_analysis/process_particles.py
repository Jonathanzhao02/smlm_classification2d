from time import time
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib
import json
import sys

from clustering import KMeansClusterIdentification, DBSCANClusterIdentification, MeanShiftClusterIdentification
from align import GridAlignment, LocalizationCluster

VALID_CLUSTER_METHODS = ['kmeans', 'dbscan', 'meanshift', 'mle']
VALID_ALIGNMENT_METHODS = ['differential_evolution', 'shgo', 'dual_annealing', 'rough']
VALID_TEMPLATE_NAMES = ['nsf', 'asu_2', 'asu_3']

# General params
DISPLAY_PARTICLE = False
DISPLAY_FINAL_CLUSTER = False
DISPLAY_DISTANCES = False
DISPLAY_OPTIMIZATION_LANDSCAPE = False
DISPLAY_GRID = False

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing clusters.mat and subParticles.mat")
    parser.add_argument("--output", "-o", help="Name of output file", default='final.mat')
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--cluster", "-c", help="Clustering method to use", default='kmeans')
    parser.add_argument("--alignment", "-a", help="Alignment optimization method to use", default='differential_evolution')
    parser.add_argument("--config", "-j", help="Path to config.json file", default=str(Path(__file__).parent.joinpath("config.json")))
    args = parser.parse_args()

    template = importlib.import_module(f".{args.template}", package="templates")

    with Path(args.config).open() as f:
        config = json.load(f)

    f = Path(args.input)
    clusters = loadmat(f.joinpath("clusters.mat"))['clusters'][0]
    subParticles = loadmat(f.joinpath("subParticles.mat"))['subParticles'][0]

    picks = np.empty(len(subParticles), dtype='O')
    datatype = np.dtype([
        ('points', 'O'),
        ('sigma', 'O'),
        ('group', 'S1024'),
        ('cluster', 'i'),
        ('raw_read', 'O'),
        ('correct', 'b'),
        ('centroids', 'O'),
        ('grid', 'O'),
        ('cost', 'f8'),
    ])

    cluster_method = args.cluster
    alignment_method = args.alignment

    if cluster_method not in VALID_CLUSTER_METHODS:
        raise Exception("Invalid cluster method! Possible values are " + ", ".join(VALID_CLUSTER_METHODS))
    
    if alignment_method not in VALID_ALIGNMENT_METHODS:
        raise Exception("Invalid alignment method! Possible values are " + ", ".join(VALID_ALIGNMENT_METHODS))
    
    if args.template not in VALID_TEMPLATE_NAMES:
        raise Exception("Invalid template! Possible values are " + ", ".join(VALID_TEMPLATE_NAMES))

    n_particles = 0
    group_avgs = np.zeros(clusters.size)
    global_avg = 0

    # Loop over each identified class
    for class_id,groups in enumerate(clusters):
        group_n_particles = groups.size
        n_particles += groups.size

        if group_n_particles == 0:
            continue

        # Loop over each particle in each class
        for group_idx,group in enumerate(groups.flatten()):
            xlim = ylim = None
            start = time()

            points = subParticles[group - 1]['points'][0][0]
            sigma = subParticles[group - 1]['sigma'][0][0][:,0]
            x = points[:,0]
            y = points[:,1]

            #if subParticles[group - 1]['group'][0][0][0][0] != 'N':
            #    continue

            if DISPLAY_PARTICLE:
                plt.figure(figsize=(6,6))
                plt.title(f'Class {class_id} Group {group}')
                plt.plot(x,y,'.')
                # xlim = plt.xlim()
                # ylim = plt.ylim()
                plt.show()
            
            cargs = config['methods'][cluster_method][args.template]

            if cluster_method != 'mle':
                if cluster_method == 'kmeans':
                    cluster = KMeansClusterIdentification(points)
                    cluster.cluster(DISPLAY_FINAL_CLUSTER, **cargs, xlim=xlim, ylim=ylim)
                elif cluster_method == 'dbscan':
                    cluster = DBSCANClusterIdentification(points)
                    cluster.cluster(DISPLAY_FINAL_CLUSTER, **cargs, xlim=xlim, ylim=ylim)
                elif cluster_method == 'meanshift':
                    print('Performing mean shift')
                    cluster = MeanShiftClusterIdentification(points)
                    cluster.cluster(DISPLAY_FINAL_CLUSTER, **cargs)

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
                    plt.plot(x,y,'.')

                    for i in range(n_clusters):
                        plt.plot(*centroids[i], 'r*')
                    
                    dist_mean = np.mean(dist_matrix[dist_matrix > 0])
                    
                    for i in range(n_clusters - 1):
                        for j in range(i+1, n_clusters):
                            if dist_matrix[i,j] < dist_mean:
                                plt.plot([centroids[i][0], centroids[j][0]], [centroids[i][1], centroids[j][1]], 'r--')
                                mid = (centroids[i] + centroids[j]) / 2
                                plt.text(*mid, str(dist_matrix[i,j])[:5], ha='center', va='center')
                
                    # plt.xlim(xlim)
                    # plt.ylim(ylim)
                    plt.show()
                
                aargs = config['alignment_optimizers'][alignment_method]
                alignment = GridAlignment(template.GRID, centroids, template.GRID_WEIGHTS, 1. / cluster.cluster_sizes, template.ORIENTATION_IDXES)
                print('Calculating transform')
                cost, tr = alignment.align(template.BOUNDS, method=alignment_method, method_args=aargs)
                print(cost, tr)

                if DISPLAY_OPTIMIZATION_LANDSCAPE:
                    alignment.plot_landscape(template.BOUNDS)

                nn = np.unique(alignment.nn[1]) # read out binary
                raw_read = np.zeros(template.GRID.shape[0])
                raw_read[nn] = 1

                end = time()

                # Only for display purposes, can ignore
                if DISPLAY_GRID:
                    gridTran = alignment.gridTran
                    inv_nn = np.setdiff1d(np.arange(GRID.shape[0]), nn, True)
                    plt.figure(figsize=(6,6))
                    plt.title(f'Class {class_id} Aligned Template')
                    plt.plot(x,y,'.')
                    c_handle = plt.plot(centroids[:,0], centroids[:,1], 'k+', markersize=12, label='Centroids')[0]
                    g0_handle = plt.plot(gridTran[inv_nn,0], gridTran[inv_nn,1], 'k.', markersize=12, label='Template (0)')[0]
                    g1_handle = plt.plot(gridTran[nn,0], gridTran[nn,1], '.', markersize=12, color='#00FF00', label='Template (1)')[0]
                    # plt.xlim(xlim)
                    # plt.ylim(ylim)
                    plt.legend(handles=[c_handle, g0_handle, g1_handle])
                    plt.show()
            
            else:
                alignment = LocalizationCluster(template.GRID, np.array([x, y, sigma]).T, cargs['bandwidth'], False)
                print('Calculating transform')
                cost, _ = alignment.fitAndPixelate((template.GRID.shape[0],1), template.SCALE, 5000, 1e-6)
                hist = alignment.hist[:-1]
                mean_size = np.mean(np.sort(hist)[::-1][:cargs['top_n_clusters']])
                size_thresh = mean_size * cargs['size_threshold']
                hist[hist < size_thresh] = 0
                hist[hist >= size_thresh] = 1
                raw_read = hist.astype(int)

                end = time()
                
                if DISPLAY_GRID:
                    gridTran = alignment.gridTran
                    plt.figure(figsize=(6,6))
                    plt.title(f'Class {class_id} Aligned Template')
                    plt.scatter(x,y,s=np.ones(x.size),alpha=0.3)
                    plt.scatter(gridTran[:,0], gridTran[:,1], c=hist)
                    plt.show()
                
                centroids = []

            # Record results
            global_avg += end - start
            group_avgs[class_id] += end - start

            picks[group - 1] = np.array([(
                points,
                subParticles[group - 1]['sigma'][0][0],
                subParticles[group - 1]['group'][0][0][0],
                class_id,
                raw_read,
                template.read_match(raw_read, subParticles[group - 1]['group'][0][0][0]),
                centroids,
                alignment.gridTran,
                cost,
            )], dtype=datatype)
        
        group_avgs[class_id] /= group_n_particles
    
    global_avg /= n_particles

    print("Avg. Times:")
    print(group_avgs)
    print(global_avg)

    # Save all read results
    savemat(str(f.joinpath(args.output)), { 'picks': picks })
