from time import time
from scipy.io import loadmat, savemat
from scipy.spatial import ConvexHull
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib
import json
import math
import sys
import csv

from align import GridAlignment, LocalizationCluster

SHOW_GFX = True
SHOW_FINAL = True

VALID_ALIGNMENT_METHODS = ['differential_evolution', 'shgo', 'dual_annealing', 'rough', 'finetune']
VALID_TEMPLATE_NAMES = ['asu_2', 'asu_3']

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

# ......
# From: https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

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

def process_clusters_hull(args):
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

    # Loop over each particle
    for idx in range(n_particles):
        clusters = pickClusters[idx]
        group = clusters['group'][0]

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

        # nn = np.unique(alignment.nn[1]) # read out binary
        # raw_read = np.zeros(template.GRID.shape[0])
        # raw_read[nn] = points

        #hull = ConvexHull(centroids)
        #plt.plot(centroids[hull.vertices,0], centroids[hull.vertices,1], 'r--', lw=2)

        # points = clusters['points']
        # x = points[:,0]
        # y = points[:,1]
        # plt.plot(x, y, '.', alpha=0.2)

        # x = centroids[:,0]
        # y = centroids[:,1]
        # plt.plot(x, y, 'ko')

        # x = alignment.gridTran[:,0]
        # y = alignment.gridTran[:,1]
        # plt.plot(x, y, '*', color='#00FF00', alpha=0.8)

        bbox = np.array(minimum_bounding_rectangle(centroids))
        dx = (np.min(bbox[:,0]) + np.max(bbox[:,0])) / 2.
        dy = (np.min(bbox[:,1]) + np.max(bbox[:,1])) / 2.
        bbox[:,0] -= dx
        bbox[:,1] -= dy

        A = bbox[0]
        B = bbox[1]
        C = bbox[2]
        D = bbox[3]

        ab = np.linalg.norm(A - B)
        bc = np.linalg.norm(B - C)

        if ab > bc:
            v = A - B
        else:
            v = C - B
        
        theta = -math.atan(v[1] / v[0])
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        bbox = np.dot(rot, bbox.T).T

        x1 = abs(bbox[0][0])
        y1 = abs(bbox[0][1])
        x2 = abs(bbox[1][0])
        y2 = abs(bbox[1][1])
        alpha = (x1 * x1 - x2 * x2) / (2 * x1 * y1 + 2 * x2 * y2)
        shearX = np.array([[1, alpha], [0, 1]])
        bbox = np.dot(shearX, bbox.T).T

        A = bbox[0]
        B = bbox[1]
        C = bbox[2]

        ab = np.linalg.norm(A - B)
        bc = np.linalg.norm(B - C)

        width = np.max(template.GRID[:,0]) * 2
        height = np.max(template.GRID[:,1]) * 2

        if ab > bc:
            scaleX = width / ab
            scaleY = height / bc
        else:
            scaleX = width / bc
            scaleY = height / ab
        
        scaleMat = np.array([[scaleX, 0], [0, scaleY]])
        bbox = np.dot(scaleMat, bbox.T).T

        if SHOW_GFX:
            plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)

        points = clusters['points']
        points[:,0] -= dx
        points[:,1] -= dy
        points = np.dot(scaleMat, np.dot(shearX, np.dot(rot, points.T))).T
        clusters['points'] = points

        points = centroids
        points[:,0] -= dx
        points[:,1] -= dy
        points = np.dot(scaleMat, np.dot(shearX, np.dot(rot, points.T))).T
        centroids = points

        alignment = GridAlignment(template.GRID, centroids, template.GRID_WEIGHTS, cluster_sizes, template.ORIENTATION_IDXES)

        d1 = alignment.squareNNDist([0,0,1,1,0])
        d2 = alignment.squareNNDist([0,0,1,1,np.pi])

        if d1 > d2:
            theta = np.pi
        else:
            theta = 0
        
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

        points = clusters['points']
        points = np.dot(rot, points.T).T
        clusters['points'] = points

        if SHOW_GFX:
            x = points[:,0]
            y = points[:,1]
            plt.plot(x, y, '.', alpha=0.2)

        points = centroids
        points = np.dot(rot, points.T).T
        centroids = points

        if SHOW_GFX:
            x = points[:,0]
            y = points[:,1]
            plt.plot(x, y, 'ko')

        points = template.GRID

        if SHOW_GFX:
            x = points[:,0]
            y = points[:,1]
            plt.plot(x, y, '*', color='#00FF00')

            plt.show()
        
        print('Calculating transform')
        alignment = GridAlignment(template.GRID, centroids, template.GRID_WEIGHTS, cluster_sizes, template.ORIENTATION_IDXES)
        aargs = config['alignment_optimizers'][alignment_method]
        cost, tr = alignment.align(template.BOUNDS, method=alignment_method, method_args=aargs)
        print(cost, tr)
        transform = tr

        nn = np.unique(alignment.nn[1]) # read out binary
        raw_read = np.zeros(template.GRID.shape[0])
        raw_read[nn] = 1

        end = time()

        if SHOW_FINAL:
            points = clusters['points']
            x = points[:,0]
            y = points[:,1]
            gridTran = alignment.gridTran
            inv_nn = np.setdiff1d(np.arange(template.GRID.shape[0]), nn, True)
            plt.figure(figsize=(6,6))
            plt.title(f'Aligned Template')
            plt.plot(x,y,'.')
            c_handle = plt.plot(centroids[:,0], centroids[:,1], 'k+', markersize=12, label='Centroids')[0]
            g0_handle = plt.plot(gridTran[inv_nn,0], gridTran[inv_nn,1], 'k.', markersize=12, label='Template (0)')[0]
            g1_handle = plt.plot(gridTran[nn,0], gridTran[nn,1], '.', markersize=12, color='#00FF00', label='Template (1)')[0]
            # plt.xlim(xlim)
            # plt.ylim(ylim)
            plt.legend(handles=[c_handle, g0_handle, g1_handle])
            plt.show()

        # Record results
        global_avg += end - start

        correct_read = template.read_match(raw_read, group)

        if correct_read:
            correct += 1

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

    # savemat(str(Path(args.input).joinpath(args.output)), { 'picks': picks, 'config': config, 'args': vars(args) })

    print(correct, n_particles)

    return correct, n_particles

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing subParticles.mat")
    parser.add_argument("--infile", "-i", help="Name of input clusters file", default='clusters.mat')
    parser.add_argument("--output", "-o", help="Name of output file", default='final.mat')
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--alignment", "-a", help="Alignment optimization method to use", default='differential_evolution')
    parser.add_argument("--config", "-j", help="Path to config.json file", default=str(Path(__file__).parent.joinpath("config.json")))
    args = parser.parse_args()

    process_clusters_hull(args)
