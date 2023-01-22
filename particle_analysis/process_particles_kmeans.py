from time import time
from sklearn.cluster import KMeans
from scipy.io import loadmat, savemat
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import json
import sys

'''
This script performs K-means clustering over a set of origami and saves all results to a file
'''

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

class KMeansClusterer():
    def __init__(self, points):
        self.points = points
    
    def optimize_clusters(self, class_sweep=[3,13], **kwargs):
        class_sweep = list(range(*class_sweep))
        cluster_results = np.empty(len(class_sweep), dtype=cluster_results_dtype)
        inertias = np.zeros(len(class_sweep), dtype=np.float64)

        for i in tqdm(class_sweep):
            idx = i - class_sweep[0]
            cluster_ids, inertia = self._cluster(i, **kwargs)
            cluster_results[idx]['n_clusters'] = i
            cluster_results[idx]['inertia'] = inertias[idx] = inertia
            cluster_results[idx]['centroid_assignments'] = cluster_ids
            cluster_results[idx]['cluster_sizes'] = np.zeros((i,), dtype=int)
            cluster_results[idx]['centroids'] = np.zeros((i,2))
            
            for j in range(i):
                ids = cluster_ids == j
                cluster_results[idx]['cluster_sizes'][j] = np.sum(ids)
                cluster_results[idx]['centroids'][j] = np.mean(self.points[ids], axis=0)
        
        d_inertias = np.diff(inertias, n=1)
        d_inertias = -d_inertias / np.max(abs(d_inertias))

        return cluster_results, d_inertias
    
    def _cluster(self, n_clusters, **kwargs):
        model = KMeans(n_clusters, **kwargs)
        cluster_ids = model.fit_predict(self.points)
        return cluster_ids, model.inertia_

def process_particles_kmeans_nocluster(args):
    with Path(args.config).open() as f:
        config = json.load(f)

    f = Path(args.input)
    subParticles = loadmat(f.joinpath("subParticles.mat"))['subParticles'][0]
    picks = np.empty(len(subParticles), dtype=datatype)
    cluster_method = 'kmeans'

    if args.template not in VALID_TEMPLATE_NAMES:
        raise Exception("Invalid template! Possible values are " + ", ".join(VALID_TEMPLATE_NAMES))

    n_particles = subParticles.size
    global_avg = 0

    # Loop over each particle
    for idx in range(n_particles):
        start = time()

        points = subParticles[idx]['points'][0][0]
        sigma = subParticles[idx]['sigma'][0][0][:,0]
        
        cargs = config['methods'][cluster_method][args.template]

        cluster = KMeansClusterer(points)
        cluster_results, inertia_curve = cluster.optimize_clusters(cargs['class_sweep'])

        end = time()

        # Record results
        global_avg += end - start

        picks[idx] = (
            points,
            subParticles[idx]['sigma'][0][0],
            subParticles[idx]['group'][0][0][0],
            cluster_results,
            inertia_curve,
        )
    
    global_avg /= n_particles

    print("Avg. Times:")
    print(global_avg)

    # Save all read results
    savemat(str(f.joinpath(args.output)), { 'picks': picks, 'config': config, 'args': vars(args) })

    return

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing subParticles.mat")
    parser.add_argument("--output", "-o", help="Name of output file", default='clusters.mat')
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--config", "-j", help="Path to config.json file", default=str(Path(__file__).parent.joinpath("config.json")))
    args = parser.parse_args()

    process_particles_kmeans_nocluster(args)
