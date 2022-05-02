from scipy.io import loadmat
from scipy.spatial import cKDTree
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib
import sys

# General params
DISPLAY_HISTOGRAM = True
DISPLAY_TEMPLATES = False
DISPLAY_BIT_ERRORS = False

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing readout results")
    parser.add_argument("--infile", "-i", help="Name of readout results file", default='final.mat')
    parser.add_argument("--output", "-o", help="Name of output folder", default="final")
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    args = parser.parse_args()

    template = importlib.import_module(f".{args.template}", package="templates")

    f = Path(args.input)
    out_f = f.joinpath(args.output)
    out_f.mkdir(exist_ok=True)

    picks = loadmat(f.joinpath(args.infile))['picks'][0]

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

    # Filter reads based on cost value
    filt_idx = costs < template.DIST_THRESH

    n_picks = np.sum(filt_idx)
    print(f"Filtered out {picks.size - n_picks} / {picks.size}, {100 * (picks.size - n_picks) / picks.size:.2f}%")

    distr = {}

    for i in np.arange(picks.size)[np.logical_not(filt_idx)]:
        v = template.to_string(template.readout(template.true_read(bytes.decode(groups[i]))))
        if v in distr:
            distr[v] += 1
        else:
            distr[v] = 1
    
    print("Filtered distribution (based on raw read):")
    print(distr)

    raw_reads = raw_reads[filt_idx]
    correct_reads = correct_reads[filt_idx]
    groups = groups[filt_idx]
    points = points[filt_idx]
    centroids = centroids[filt_idx]
    grids = grids[filt_idx]
    costs = costs[filt_idx]
    clusters = clusters[filt_idx]

    # Calculate sizes of each cluster
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        clusters_sizes[i] = np.sum(clusters == i)

    # Tracks which template positions were read in each cluster
    group_counts = np.zeros((n_clusters, template.GRID.shape[0]), dtype=int)

    # Tracks what readouts there are
    readouts = np.empty(n_clusters, dtype='O')
    for i in range(n_clusters):
        readouts[i] = np.zeros(clusters_sizes[i], dtype=int)
    
    # Tracks index of clusters while inserting readouts
    inds = np.zeros(n_clusters, dtype=int)

    # Bit-error specific tracking
    distance_errors = {}
    bit_fp_errors = {}
    bit_fn_errors = {}
    bin_counts = {}

    for i in range(n_picks):
        read = raw_reads[i]
        cluster = clusters[i]
        correct = correct_reads[i]
        group = bytes.decode(groups[i])
        target = template.true_read(group)
        g_bin = template.binnify(group)

        grid = grids[i]
        centroid = centroids[i]

        # Check orientation markers are all present
        if np.sum(read[template.ORIENTATION_IDXES]) == template.ORIENTATION_IDXES.size:
            group_counts[cluster] += read

        if g_bin not in distance_errors:
            distance_errors[g_bin] = np.zeros(template.GRID.shape[0], dtype=float)
            bit_fp_errors[g_bin] = np.zeros(template.GRID.shape[0], dtype=int)
            bit_fn_errors[g_bin] = np.zeros(template.GRID.shape[0], dtype=int)
            bin_counts[g_bin] = np.zeros(template.GRID.shape[0], dtype=int)

        if centroid.size > 0:
            nnTree = cKDTree(grid)
            [dist, nn] = nnTree.query(centroid)

            for j,n_idx in enumerate(nn):
                distance_errors[g_bin][n_idx] += dist[j]

        if target:
            bin_counts[g_bin] += read

            bit_errs = np.logical_xor(read, target)
            bit_fp_errs = np.logical_and(bit_errs, target == 0)
            bit_fn_errs = np.logical_and(bit_errs, target == 1)

            bit_fp_errors[g_bin] += bit_fp_errs
            bit_fn_errors[g_bin] += bit_fn_errs
                
        readouts[cluster][inds[cluster]] = template.readout(read)
        inds[cluster] += 1
    
    for key,val in distance_errors.items():
        distance_errors[key] = np.divide(val, bin_counts[g_bin])
    
    # Display histogram of overall and individual cluster reads
    if DISPLAY_HISTOGRAM:
        for i in range(n_clusters):
            plt.title(f"Class {i}")
            [n, bins, _] = plt.hist(readouts[i], bins=range(1 << template.REPETITION_PAIRS.shape[0]))
            plt.ylim(top=clusters_sizes[i])
            plt.show()

            top_n = bins[np.argsort(n)[::-1][:n_clusters]].astype(int)
            print(np.sort(n)[::-1][:n_clusters])
            print(list(map(template.to_string, top_n)))
        
        all_readouts = np.concatenate(readouts)
        plt.title("Overall")
        [n, bins, _] = plt.hist(all_readouts, bins=range(1 << template.REPETITION_PAIRS.shape[0]))
        plt.ylim(top=n_picks)
        plt.show()

        top_n = bins[np.argsort(n)[::-1][:n_clusters]].astype(int)
        print(np.sort(n)[::-1][:n_clusters])
        print(list(map(template.to_string, top_n)))

    # Create visual representation of counts at each template position
    if DISPLAY_TEMPLATES:
        for class_id in range(n_clusters):
            plt.title(f"Class {class_id}")
            counts = group_counts[class_id]

            for i,point in enumerate(template.GRID):
                plt.text(*point, str(counts[i]), ha='center', va='center')

            x = template.GRID[:,0]
            y = template.GRID[:,1]

            plt.xlim(x.min() * 1.5,x.max() * 1.5)
            plt.ylim(y.min() * 1.5,y.max() * 1.5)

            plt.show()
    
    if DISPLAY_BIT_ERRORS:
        for key in distance_errors.keys():
            plt.figure(figsize=(8,8))

            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, 3)

            ax1.set_title(f"Class {key} False Negatives")
            ax2.set_title(f"Class {key} False Positives")
            ax3.set_title(f"Class {key} Distances")
            
            for i,point in enumerate(template.GRID):
                ax1.text(*point, str(bit_fn_errors[key][i]), ha='center', va='center')
                ax2.text(*point, str(bit_fp_errors[key][i]), ha='center', va='center')
                ax3.text(*point, f'{distance_errors[key][i]:.2e}', ha='center', va='center')
            
            x = template.GRID[:,0]
            y = template.GRID[:,1]

            ax1.set_xlim(x.min() * 1.5,x.max() * 1.5)
            ax1.set_ylim(y.min() * 1.5,y.max() * 1.5)
            ax2.set_xlim(x.min() * 1.5,x.max() * 1.5)
            ax2.set_ylim(y.min() * 1.5,y.max() * 1.5)
            ax3.set_xlim(x.min() * 1.5,x.max() * 1.5)
            ax3.set_ylim(y.min() * 1.5,y.max() * 1.5)

            plt.show()
    
    print(f"Correctly read {correct_reads.sum()} / {n_picks} picks, {correct_reads.sum() / n_picks * 100:.2f}%")

    # Saves images of misclassified particles
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
        pick_points = points_mis[i]
        pick_centroids = centroids_mis[i]
        pick_grid = grids_mis[i]
        pick_read = raw_reads_mis[i].astype(bool)
        pick_cost = costs_mis[i]

        pick_val = template.readout(pick_read)

        inv_read = np.logical_not(pick_read)
        plt.figure(figsize=(6,6))
        plt.title(f'Pick {pick_group} Aligned Template, Read {template.to_string(pick_val)}, Cost {pick_cost:.3e}')
        plt.plot(pick_points[:,0],pick_points[:,1],',')
        if pick_centroids.size > 0:
            plt.plot(pick_centroids[:,0], pick_centroids[:,1], 'r*')
        plt.plot(pick_grid[inv_read,0], pick_grid[inv_read,1], 'k*')
        plt.plot(pick_grid[pick_read,0], pick_grid[pick_read,1], '*', color='#00FF00')
        plt.savefig(out_path.joinpath(f"{pick_group}.png"))
        plt.close()

# TODO:
# 3. 2 and 3 repetition again w/ labeled
# inf. show bima + ritvik process

# -1. add config file
# 0. bit errors location-wise or #
# 1. binding-site-wise distance map
# 2. run MLE on NSF + meanshift
# 4. elbow on KMeans on MDS (NSF)
# 5. run MLE/meanshift/K-means over superparticles

# crytographic hash function
