from sklearn.cluster import KMeans, DBSCAN, MeanShift
import matplotlib.pyplot as plt
import numpy as np

from tqdm.auto import tqdm

class KMeansClusterIdentification():
    def __init__(self, points):
        self.points = points
        self.n_clusters = 0
    
    def optimize_clusters(self, class_sweep=[3,13], elbow_threshold=0.15, **kwargs):
        inertias = []
        class_sweep = list(range(*class_sweep))

        for i in tqdm(class_sweep):
            cluster_ids, inertia = self._cluster(i, **kwargs)
            inertias.append(inertia)
        
        d_inertias = np.diff(inertias, n=1)
        d_inertias = -d_inertias / np.max(abs(d_inertias))
        dists = np.abs(d_inertias - elbow_threshold)
        self.n_clusters = class_sweep[:-1][np.argmin(dists)]
    
    def _cluster(self, n_clusters, **kwargs):
        model = KMeans(n_clusters, **kwargs)
        cluster_ids = model.fit_predict(self.points)
        return cluster_ids, model.inertia_
    
    def cluster(self, display=False, size_threshold=0.8, class_sweep=[3,13], elbow_threshold=0.15, xlim=None, ylim=None, **kwargs):
        if self.n_clusters == 0:
            self.optimize_clusters(class_sweep, elbow_threshold, **kwargs)
        
        self.centroids = np.zeros((self.n_clusters,2))
        self.cluster_sizes = np.zeros(self.n_clusters)
        self.cluster_ids, _ = self._cluster(self.n_clusters, **kwargs)

        for i in range(self.n_clusters):
            ids = self.cluster_ids == i
            self.centroids[i] = np.mean(self.points[ids], axis=0)
            self.cluster_sizes[i] = np.sum(ids)

        mean_size = np.mean(self.cluster_sizes)
        size_thresh = mean_size * size_threshold
        idxes = np.arange(self.n_clusters)
        filtered_idxes = idxes[self.cluster_sizes < size_thresh][::-1]
        idxes = np.setdiff1d(idxes, filtered_idxes, True)

        for i in filtered_idxes:
            self.cluster_ids[self.cluster_ids == i] = -1

        for i in filtered_idxes:
            self.cluster_ids[self.cluster_ids > i] -= 1

        self.centroids = self.centroids[idxes]
        self.cluster_sizes = self.cluster_sizes[idxes]
        self.n_clusters -= filtered_idxes.size

        if display:
            x = self.points[:,0]
            y = self.points[:,1]

            x_means = self.centroids[:,0]
            y_means = self.centroids[:,1]

            plt.figure(figsize=(6,6))
            plt.title(f'Clusters {self.n_clusters}')

            for j in range(self.n_clusters):
                ids = self.cluster_ids == j
                plt.plot(x[ids], y[ids], '.')
            
            c_handle = plt.plot(x_means, y_means, 'k+', markersize=12, label='centroids')[0]
            
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend(handles=[c_handle])
            plt.show()

class DBSCANClusterIdentification():
    def __init__(self, points):
        self.points = points
    
    def cluster(self, display=False, xlim=None, ylim=None, **kwargs):
        model = DBSCAN(eps=8e-3, min_samples=40, n_jobs=-1, **kwargs)
        self.cluster_ids = model.fit_predict(self.points)
        self.n_clusters = np.max(self.cluster_ids) + 1
        self.centroids = np.zeros((self.n_clusters,2))
        self.cluster_sizes = np.zeros(self.n_clusters)

        for i in range(self.n_clusters):
            ids = self.cluster_ids == i
            self.centroids[i] = np.mean(self.points[ids], axis=0)
            self.cluster_sizes[i] = np.sum(ids)

        if display:
            x = self.points[:,0]
            y = self.points[:,1]

            x_means = self.centroids[:,0]
            y_means = self.centroids[:,1]

            plt.figure(figsize=(6,6))
            plt.title(f'Clusters {self.n_clusters}')

            for i in range(self.n_clusters):
                ids = self.cluster_ids == i
                plt.plot(x[ids], y[ids], ',')
                plt.plot(x_means[i], y_means[i], 'r*')
            
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.show()

class MeanShiftClusterIdentification():
    def __init__(self, points):
        self.points = points
    
    def cluster(self, display=False, bandwidth=None, top_n_clusters=3, size_threshold=0.8, xlim=None, ylim=None, **kwargs):
        model = MeanShift(bandwidth=bandwidth, n_jobs=-1, **kwargs)
        self.cluster_ids = model.fit_predict(self.points)
        self.n_clusters = np.max(self.cluster_ids) + 1
        self.centroids = np.zeros((self.n_clusters,2))
        self.cluster_sizes = np.zeros(self.n_clusters)

        for i in range(self.n_clusters):
            ids = self.cluster_ids == i
            self.centroids[i] = np.mean(self.points[ids], axis=0)
            self.cluster_sizes[i] = np.sum(ids)
        
        large_cluster_sizes = np.sort(self.cluster_sizes)[-top_n_clusters:]
        mean_size = np.mean(large_cluster_sizes)
        size_thresh = mean_size * size_threshold
        idxes = np.arange(self.n_clusters)
        filtered_idxes = idxes[self.cluster_sizes < size_thresh][::-1]
        idxes = np.setdiff1d(idxes, filtered_idxes, True)

        for i in filtered_idxes:
            self.cluster_ids[self.cluster_ids == i] = -1

        for i in filtered_idxes:
            self.cluster_ids[self.cluster_ids > i] -= 1

        self.centroids = self.centroids[idxes]
        self.cluster_sizes = self.cluster_sizes[idxes]
        self.n_clusters -= filtered_idxes.size

        if display:
            x = self.points[:,0]
            y = self.points[:,1]

            x_means = self.centroids[:,0]
            y_means = self.centroids[:,1]

            plt.figure(figsize=(6,6))
            plt.title(f'Clusters {self.n_clusters}')

            for i in range(self.n_clusters):
                ids = self.cluster_ids == i
                plt.plot(x[ids], y[ids], ',')
                plt.plot(x_means[i], y_means[i], 'r*')
            
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.show()

