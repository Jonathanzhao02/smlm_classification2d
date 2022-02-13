#!/usr/local/Caskroom/miniconda/base/envs/picasso/bin/python3
import numpy as np
from scipy.io import savemat, loadmat
from pathlib import Path
from tqdm.auto import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder of .mat files to merge")
    parser.add_argument("output", help="Name of folder to output to in data")
    parser.add_argument("--picks", "-p", help="Number of picks to draw from each file", type=int, default=0)
    args = parser.parse_args()
    
    f = Path(args.input)
    o = Path(args.output)
    n_picks = args.picks

    if not f.is_dir():
        exit()
    
    picks = None

    for file in f.iterdir():
        m = loadmat(file)['subParticles']

        if n_picks > 0:
            m = np.random.choice(m.flatten(), min(n_picks, m.size), False).reshape((1, -1))

        if picks is None:
            picks = m
        else:
            picks = np.concatenate((picks, m), axis=-1)

    out = Path(__file__).parent.joinpath(Path('../data').joinpath(o))
    out.mkdir(exist_ok=True)

    savemat(str(out.joinpath(Path('subParticles.mat'))), { 'subParticles': picks })

    # HDF5 NOTES:
    # contains 'locs', which is a Dataset object
    # datatype contains frame, x, y, photons, sx, sy, bg, lpx, lpy, ellipcity, net_gradient, group, most are <f4 (32-bit), some integers/unsigned (frame + group)

    # x/y are positions, sx/sy are uncertainties, lpx/lpy are cramer rao lower bounds to gauss maximum likelihood estimation

    # MAT NOTES:
    # dict that has irrelevant metadata info, then subParticles 1xn array, dtype='O'
    # each subParticles entry has 1x1 dict, points + sigma, dtype = [('points', 'O'), ('sigma', 'O')]
    # to create, datatype = np.dtype([('points', 'O'), ('sigma', 'O')]), then array = np.array(data, dtype=datatype) where data is organized into pairs (points, sigma)
    # to access, subParticle['points'] or subParticle['sigma']

    # points = 1x1 array, dtype='O'
    # contains mx2 array, dtype='<f8'
    # contains 2D coords of each localization within pick in camera pixel units

    # sigma = 1x1 array, dtype='O'
    # contains mx1 array, dtype='<f8'
    # contains squared uncertainties of each localization within pick in squared camera pixel units
