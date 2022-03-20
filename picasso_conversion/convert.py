import h5py as h5
import numpy as np
from scipy.io import savemat
from pathlib import Path
from tqdm.auto import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input hdf5 file to convert")
    parser.add_argument("--out", "-o", help="Name of output folder", default="")
    parser.add_argument("--tag", "-t", help="Tag string to add before group number", default="")
    parser.add_argument("--picks", "-p", help="Number of picks to draw from each file", type=int, default=0)
    args = parser.parse_args()
    
    n_picks = args.picks
    f = Path(args.file)
    data = h5.File(f, 'r')

    locs = data['locs']
    groups = np.unique(locs['group'])

    picks = np.empty(len(groups), dtype='O')
    datatype = np.dtype([('points', 'O'), ('sigma', 'O'), ('group', 'S1024')])

    points = np.stack((locs['x'], locs['y']), axis=-1).astype('<f8')
    sigma = np.linalg.norm((locs['lpx'], locs['lpy']), axis=0).astype('<f8')

    if n_picks > 0:
        groups = np.random.choice(groups, min(n_picks, groups.size), False)

    for i in tqdm(groups):
        idx = locs['group'] == i

        if np.mean(sigma[idx]) > 1:
            continue

        picks[i] = np.array([(points[idx] - np.mean(points[idx], axis=0), sigma[idx].reshape((-1, 1)), f'{args.tag}{i}')], dtype=datatype)

    picks = picks[picks != None]

    out = Path(__file__).parent.joinpath(Path('../data').joinpath(args.out or Path(f.with_suffix('').name)))
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
