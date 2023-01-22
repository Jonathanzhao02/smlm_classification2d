from time import time
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib
import json
import sys
import csv

from align import GridAlignment, LocalizationCluster

'''
This script processes the K-means results to find the optimal number of clusters and perform alignment
'''

VALID_ALIGNMENT_METHODS = ['differential_evolution', 'shgo', 'dual_annealing', 'rough']
VALID_TEMPLATE_NAMES = ['nsf', 'asu_2', 'asu_3']

cluster_results_dtype = np.dtype([
    ('inertia', np.float64),
    ('n_clusters', np.int64),
    ('cluster_sizes', 'O'),
    ('centroids', 'O'),
    ('centroid_assignments', 'O'),
])

datatype = np.dtype([
    ('points', 'O'),
    ('sigma', 'O'),
    ('group', 'S1024'),
    ('cluster_results', 'O'),
    ('inertia_curve', 'O'),
])

def filter_by_inertia(clusters, cargs):
    d_inertias = clusters['inertia_curve']
    dists = np.abs(d_inertias - cargs['elbow_threshold'])
    idx = np.argmin(dists)
    return clusters['cluster_results'][0][idx]

def filter_by_size(centroids, n_clusters, cluster_sizes, centroid_assignments, cargs):
    mean_size = np.mean(cluster_sizes)
    size_thresh = mean_size * cargs['size_threshold']
    idxes = np.arange(n_clusters)
    filtered_idxes = idxes[cluster_sizes < size_thresh][::-1]
    idxes = np.setdiff1d(idxes, filtered_idxes, True)

    for i in filtered_idxes:
        centroid_assignments[centroid_assignments == i] = -1

    for i in filtered_idxes:
        centroid_assignments[centroid_assignments > i] -= 1

    centroids = centroids[idxes]
    cluster_sizes = cluster_sizes[idxes]
    n_clusters -= filtered_idxes.size

    return centroids, n_clusters, cluster_sizes, centroid_assignments

def plot(points, centroids, centroid_assignments, grid, pick_read, cost, template, group, fname):
    n_clusters = len(centroids)
    x = points[:,0]
    y = points[:,1]

    x_means = centroids[:,0]
    y_means = centroids[:,1]

    pick_read = pick_read.astype(bool)
    inv_read = np.logical_not(pick_read)
    pick_val = template.readout(pick_read)

    fig = plt.figure(figsize=(6,6))
    plt.title(f'Pick {group} Aligned Template, Read {template.to_string(pick_val)}, Cost {cost:.3e}')

    plt.plot(grid[inv_read,0], grid[inv_read,1], 'k*')
    plt.plot(grid[pick_read,0], grid[pick_read,1], '*', color='#00FF00')

    for i in range(n_clusters):
        ids = centroid_assignments == i
        plt.plot(x[ids], y[ids], '.')
    
    plt.plot(x_means, y_means, 'ko', markersize=12)
    ax = fig.get_axes()[0]

    for i in range(n_clusters):
        ax.annotate(str(i), xy=(x_means[i], y_means[i]), horizontalalignment='center', verticalalignment='center', color='white', weight='heavy', fontsize='large')

    plt.savefig(fname)
    plt.close()

def process_clusters_kmeans(args, serialize):
    template = importlib.import_module(f".{args.template}", package="templates")

    with Path(args.config).open() as f:
        config = json.load(f)

    f = Path(args.input)
    pickClusters = loadmat(f.joinpath(args.infile))['picks'][0]

    picks = np.empty(len(pickClusters), dtype='O')
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
        ('transform', 'O'),
    ])

    cluster_method = 'kmeans'
    alignment_method = args.alignment
    
    if alignment_method not in VALID_ALIGNMENT_METHODS:
        raise Exception("Invalid alignment method! Possible values are " + ", ".join(VALID_ALIGNMENT_METHODS))
    
    if args.template not in VALID_TEMPLATE_NAMES:
        raise Exception("Invalid template! Possible values are " + ", ".join(VALID_TEMPLATE_NAMES))

    n_particles = pickClusters.size
    global_avg = 0
    correct = 0

    if serialize:
        field_names = ["PickID", "PickGroup", "ClusterID", "NumLocs"]
        filt_csv_data = []
        csv_data = []

        out_path = f.joinpath(args.outfolder)
        out_path.mkdir(exist_ok=True)

        out_f = out_path.joinpath('imgs')
        out_f.mkdir(exist_ok=True)

    # Loop over each particle
    for idx in range(n_particles):
        clusters = pickClusters[idx]
        
        start = time()

        cargs = config['methods'][cluster_method][args.template]
        cluster_results = filter_by_inertia(clusters, cargs)

        centroids = cluster_results['centroids']
        n_clusters = cluster_results['n_clusters'][0][0]
        cluster_sizes = cluster_results['cluster_sizes'][0]
        centroid_assignments = cluster_results['centroid_assignments'][0]

        centroids, n_clusters, cluster_sizes, centroid_assignments = filter_by_size(
            centroids,
            n_clusters,
            cluster_sizes,
            centroid_assignments,
            cargs
        )
        
        aargs = config['alignment_optimizers'][alignment_method]
        alignment = GridAlignment(template.GRID, centroids, template.GRID_WEIGHTS, cluster_sizes, template.ORIENTATION_IDXES)
        print('Calculating transform')
        cost, tr = alignment.align(template.BOUNDS, method=alignment_method, method_args=aargs)
        print(cost, tr)
        transform = tr
        nn = np.unique(alignment.nn[1]) # read out binary
        raw_read = np.zeros(template.GRID.shape[0])
        raw_read[nn] = 1

        end = time()

        # Record results
        global_avg += end - start

        group = clusters['group'][0]
        correct_read = template.read_match(raw_read, group)

        if correct_read:
            correct += 1
        
        if serialize:
            plot(clusters['points'], centroids, centroid_assignments, alignment.gridTran, raw_read, cost, template, group, str(out_f.joinpath(f'{group}.png')))
            
            for i in range(n_clusters):
                filt_csv_data.append({
                    "PickID": idx,
                    "PickGroup": group,
                    "ClusterID": i,
                    "NumLocs": cluster_sizes[i]
                })
            
            for i in range(cluster_results['n_clusters'][0][0]):
                csv_data.append({
                    "PickID": idx,
                    "PickGroup": group,
                    "ClusterID": i,
                    "NumLocs": cluster_results['cluster_sizes'][0][i]
                })

        picks[idx] = np.array([(
            clusters['points'],
            clusters['sigma'][:,0],
            group,
            0,
            raw_read,
            correct_read,
            centroids,
            alignment.gridTran,
            cost,
            np.array(transform),
        )], dtype=datatype)

    global_avg /= n_particles

    print("Avg. Times:")
    print(global_avg)

    # Save all read results
    if serialize:
        with out_path.joinpath("clusters.csv").open('w') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(csv_data)
        
        with out_path.joinpath("filt_clusters.csv").open('w') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(filt_csv_data)

    savemat(str(Path(args.input).joinpath(args.output)), { 'picks': picks, 'config': config, 'args': vars(args) })

    return correct, n_particles

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing subParticles.mat")
    parser.add_argument("--infile", "-i", help="Name of input clusters file", default='clusters.mat')
    parser.add_argument("--outfolder", "-of", help="Name of output folder", default='clusters')
    parser.add_argument("--output", "-o", help="Name of output file", default='final.mat')
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--alignment", "-a", help="Alignment optimization method to use", default='differential_evolution')
    parser.add_argument("--config", "-j", help="Path to config.json file", default=str(Path(__file__).parent.joinpath("config.json")))
    args = parser.parse_args()

    correct, n_particles = process_clusters_kmeans(args, True)
    print(f"{correct} / {n_particles}")
