from time import time
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib
import json
import sys
import csv

def calc_shear(sx, sy, dt):
    num = (sy * sy - sx * sx) * np.sin(dt) * np.cos(dt)
    den = np.sqrt(sx * sx * np.sin(dt) ** 2 + sy * sy * np.cos(dt) ** 2) * np.sqrt(sx * sx * np.cos(dt) ** 2 + sy * sy * np.sin(dt) ** 2)
    return np.arccos(num / den)

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between(u, v):
    u1 = unit_vector(u)
    v1 = unit_vector(v)
    return np.arccos(np.clip(np.dot(u1, v1), -1, 1))

def descale_grid(grid, transform):
    grid = grid.copy()
    dx, dy, sx, sy, dt = transform
    grid[:,0] -= dx
    grid[:,1] -= dy
    grid[:,0] /= sx
    grid[:,1] /= sy
    grid[:,0] += dx
    grid[:,1] += dy
    return grid

def reprocess(args):
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

    picks = loadmat(str(Path(args.input).joinpath(args.infile)))['picks'][0]

    data = []

    # Loop over each particle
    for idx in range(n_particles):
        pick = picks[idx][0][0]
        points = pick['points']
        raw_read = pick['raw_read'][0].astype(int)
        grid = pick['grid']
        group = pick['group'][0]
        transform = pick['transform'][0]
        data.append({
            'group': group,
            'cost': pick['cost'][0][0],
            'translationX': transform[0],
            'translationY': transform[1],
            'scaleX': transform[2],
            'scaleY': transform[3],
            'rotation': transform[4],
            'shear': calc_shear(transform[2], transform[3], transform[4]) - np.pi / 2,
        })

        if group == args.group:
            # invert grid
            grid = inv_grid = descale_grid(grid, transform)

        a = grid[8]
        b = grid[9]
        c = grid[1]

        ba = a - b
        bc = c - b

        print(group, angle_between(ba, bc) / 2 / np.pi * 360 - 90, calc_shear(transform[2], transform[3], transform[4]) / np.pi / 2 * 360 - 90)
    
    out_path = f.joinpath(args.outfolder)
    out_path.mkdir(exist_ok=True)
    
    field_names = ['group', 'cost', 'translationX', 'translationY', 'scaleX', 'scaleY', 'rotation', 'shear']
    with out_path.joinpath("transform.csv").open('w') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)
    
    if args.group:
        f = Path(args.input)
        out_path = f.joinpath(args.group)
        out_path.mkdir(exist_ok=True)

        field_names = ['x', 'y']
        csv_data = []

        for i in range(inv_grid.shape[0]):
            csv_data.append({
                'x': inv_grid[i][0],
                'y': inv_grid[i][1],
            })
        
        with out_path.joinpath("invgrid.csv").open('w') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(csv_data)

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing subParticles.mat")
    parser.add_argument("--group", "-g", help="Group ID of particle to retrieve inverted grid of", default=None)
    parser.add_argument("--infile", "-i", help="Name of input file", default='final.mat')
    parser.add_argument("--outfolder", "-of", help="Name of output folder", default='final')
    args = parser.parse_args()

    reprocess(args)
