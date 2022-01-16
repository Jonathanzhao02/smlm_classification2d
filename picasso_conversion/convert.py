#!/usr/local/Caskroom/miniconda/base/envs/picasso/bin/python3
import h5py as h5
import numpy as np
from scipy.io import savemat, loadmat
from pathlib import Path
from tqdm.auto import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input hdf5 file to convert")
    args = parser.parse_args()
    
    f = Path(args.file)
    data = h5.File(f, 'r')

    locs = data['locs']
    groups = np.unique(locs['group'])

    picks = [[[], []] for _ in range(len(groups))]

    for i in tqdm(range(len(locs))):
        loc = locs[i]
        group = loc['group']

        x, y = loc['x'], loc['y']
        s = np.linalg.norm([loc['sx'], loc['sy']])

        picks[group][0].append((x, y))
        picks[group][1].append(s)
    
    datatype = np.dtype([('points', 'O'), ('sigma', 'O')])
    # array = np.empty((1, len(groups)), dtype='O')
    array = np.empty(len(groups), dtype='O')

    for i in tqdm(range(len(groups))):
        pick = picks[i]

        points = np.array(pick[0]).astype('<f8')
        # sigma = np.array(pick[1]).astype('<f8')[:,np.newaxis]
        sigma = np.array(pick[1]).astype('<f8')[:,np.newaxis]

        # array[0,i] = np.array([(points, sigma)], dtype=datatype)[:,np.newaxis]
        array[i] = np.array([(points, sigma)], dtype=datatype)

    out = Path('../data').joinpath(Path(f.with_suffix('').name))
    out.mkdir(exist_ok=True)

    savemat(str(out.joinpath(Path('subParticles.mat'))), { 'subParticles': array })

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
