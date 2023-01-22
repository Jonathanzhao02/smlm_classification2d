from time import time
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib
import json
import sys
import csv

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

def plot(points, centroids, centroid_assignments, grid, pick_read, group):
    n_clusters = len(centroids)
    x = points[:,0]
    y = points[:,1]

    x_means = centroids[:,0]
    y_means = centroids[:,1]

    pick_read = pick_read.astype(bool)
    inv_read = np.logical_not(pick_read)

    fig = plt.figure(figsize=(6,6))
    plt.title(f'Pick {group} Aligned Template')

    plt.plot(grid[inv_read,0], grid[inv_read,1], 'k*')
    plt.plot(grid[pick_read,0], grid[pick_read,1], '*', color='#00FF00')

    for i in range(n_clusters):
        ids = centroid_assignments == i
        plt.plot(x[ids], y[ids], '.')
    
    plt.plot(x_means, y_means, 'ko', markersize=12)
    ax = fig.get_axes()[0]

    for i in range(n_clusters):
        ax.annotate(str(i), xy=(x_means[i], y_means[i]), horizontalalignment='center', verticalalignment='center', color='white', weight='heavy', fontsize='large')

    plt.show()

def particle_to_csv(args):
    with Path(args.config).open() as f:
        config = json.load(f)
    
    f = Path(args.input)
    pickClusters = loadmat(f.joinpath(args.infile))['picks'][0]

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

    final_datatype = np.dtype([
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

    n_particles = pickClusters.size
    global_avg = 0
    correct = 0

    picks = loadmat(str(Path(args.input).joinpath(args.output)))['picks'][0]

    # Loop over each particle
    for idx in range(n_particles):
        clusters = pickClusters[idx]
        
        cargs = config['methods']['kmeans'][args.template]
        cluster_results = filter_by_inertia(clusters, cargs)

        centroids = cluster_results['centroids']
        n_clusters = cluster_results['n_clusters'][0][0]
        cluster_sizes = cluster_results['cluster_sizes'][0]
        centroid_assignments = cluster_results['centroid_assignments'][0]

        # get final used centroids
        centroids, n_clusters, cluster_sizes, centroid_assignments = filter_by_size(
            centroids,
            n_clusters,
            cluster_sizes,
            centroid_assignments,
            cargs
        )

        group = clusters['group'][0]

        pick = picks[idx][0][0]
        points = pick['points']
        raw_read = pick['raw_read'][0].astype(int)
        grid = pick['grid']

        if group == args.group:
            break
    
    out_path = f.joinpath(group)
    out_path.mkdir(exist_ok=True)

    field_names = ['x', 'y', 'cluster']
    csv_data = []

    for i in range(points.shape[0]):
        csv_data.append({
            'x': points[i][0],
            'y': points[i][1],
            'cluster': centroid_assignments[i]
        })

    with out_path.joinpath("points.csv").open('w') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(csv_data)
    
    field_names = ['x', 'y', 'cluster']
    csv_data = []

    for i in range(centroids.shape[0]):
        csv_data.append({
            'x': centroids[i][0],
            'y': centroids[i][1],
            'cluster': i
        })

    with out_path.joinpath("centroids.csv").open('w') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(csv_data)
    
    field_names = ['x', 'y', 'on']
    csv_data = []

    for i in range(grid.shape[0]):
        csv_data.append({
            'x': grid[i][0],
            'y': grid[i][1],
            'on': raw_read[i]
        })

    with out_path.joinpath("grid.csv").open('w') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(csv_data)

# Converts a single particle's grid, centroids, and localizations to CSV form
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing subParticles.mat")
    parser.add_argument("group", help="Group ID of particle to convert")
    parser.add_argument("--infile", "-i", help="Name of input clusters file", default='clusters.mat')
    parser.add_argument("--output", "-o", help="Name of output file", default='final.mat')
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--config", "-j", help="Path to config.json file", default=str(Path(__file__).parent.joinpath("config.json")))
    args = parser.parse_args()

    particle_to_csv(args)
