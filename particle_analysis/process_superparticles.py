from scipy.io import loadmat
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
DISPLAY_PARTICLE = True
DISPLAY_FINAL_CLUSTER = True
DISPLAY_DISTANCES = False
DISPLAY_GRID = True

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing classes.mat")
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--cluster", "-c", help="Clustering method to use", default='kmeans')
    parser.add_argument("--alignment", "-a", help="Alignment optimization method to use", default='differential_evolution')
    parser.add_argument("--config", "-j", help="Path to config.json file", default=str(Path(__file__).parent.joinpath("config_superparticle.json")))
    args = parser.parse_args()

    template = importlib.import_module(f".{args.template}", package="templates")

    with Path(args.config).open() as f:
        config = json.load(f)

    f = Path(args.input).joinpath("classes.mat")
    classes = loadmat(f)['classes'][0]

    cluster_method = args.cluster
    alignment_method = args.alignment

    if cluster_method not in VALID_CLUSTER_METHODS:
        raise Exception("Invalid cluster method! Possible values are " + ", ".join(VALID_CLUSTER_METHODS))
    
    if alignment_method not in VALID_ALIGNMENT_METHODS:
        raise Exception("Invalid alignment method! Possible values are " + ", ".join(VALID_ALIGNMENT_METHODS))
    
    if args.template not in VALID_TEMPLATE_NAMES:
        raise Exception("Invalid template! Possible values are " + ", ".join(VALID_TEMPLATE_NAMES))

    # Loop over each identified superparticle class
    for class_id in range(len(classes)):
        points = classes[class_id][0,-1]
        x = points[:,0]
        y = points[:,1]

        if DISPLAY_PARTICLE:
            plt.figure(figsize=(6,6))
            plt.title(f'Class {class_id}')
            plt.plot(x,y,',')
            plt.show()
        
        cargs = config['methods'][cluster_method][args.template]

        if cluster_method != 'mle':
            if cluster_method == 'kmeans':
                cluster = KMeansClusterIdentification(points)
                cluster.cluster(DISPLAY_FINAL_CLUSTER, **cargs)
            elif cluster_method == 'dbscan':
                cluster = DBSCANClusterIdentification(points)
                cluster.cluster(DISPLAY_FINAL_CLUSTER, **cargs)
            elif cluster_method == 'meanshift':
                print('Performing mean shift')
                cluster = MeanShiftClusterIdentification(points)
                cluster.cluster(DISPLAY_FINAL_CLUSTER, **cargs)

            n_clusters = cluster.n_clusters
            centroids = cluster.centroids
            
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
            
            aargs = config['alignment_optimizers'][alignment_method]
            alignment = GridAlignment(template.GRID, centroids, template.GRID_WEIGHTS, 1. / cluster.cluster_sizes, template.ORIENTATION_IDXES)
            print('Calculating transform')
            #cost, tr = alignment.align(BOUNDS, method='rough', method_args={ 'gridsize': SCALE / 4., 'steps': 8 })
            cost, tr = alignment.align(template.BOUNDS, method=alignment_method, method_args=aargs)
            print(cost, tr)

            if DISPLAY_GRID:
                gridTran = alignment.gridTran
                nn = np.unique(alignment.nn[1])
                inv_nn = np.setdiff1d(np.arange(template.GRID.shape[0]), nn, True)
                plt.figure(figsize=(6,6))
                plt.title(f'Class {class_id} Aligned Template')
                plt.plot(x,y,',')
                plt.plot(centroids[:,0], centroids[:,1], 'r*')
                plt.plot(gridTran[inv_nn,0], gridTran[inv_nn,1], 'k*')
                plt.plot(gridTran[nn,0], gridTran[nn,1], '*', color='#00FF00')
                plt.show()
        else:
            sigma = np.zeros(x.size)
            sigma[:] = cargs['bandwidth']

            alignment = LocalizationCluster(template.GRID, np.array([x, y, sigma]).T, cargs['bandwidth'], False)
            print('Calculating transform')
            cost, _ = alignment.fitAndPixelate((template.GRID.shape[0],1), template.SCALE, 5000, 1e-6)
            hist = alignment.hist[:-1]
            mean_size = np.mean(np.sort(hist)[::-1][:cargs['top_n_clusters']])
            size_thresh = mean_size * cargs['size_threshold']
            hist[hist < size_thresh] = 0
            hist[hist >= size_thresh] = 1

            if DISPLAY_GRID:
                gridTran = alignment.gridTran
                plt.figure(figsize=(6,6))
                plt.title(f'Class {class_id} Aligned Template')
                plt.scatter(x,y,s=np.ones(x.size),alpha=0.3)
                plt.scatter(gridTran[:,0], gridTran[:,1], c=hist)
                plt.show()
