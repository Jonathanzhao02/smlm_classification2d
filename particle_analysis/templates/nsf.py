import numpy as np

# TEMPLATE 1: 3x3 grid w/ 3 offset orientation markers
# 1 2
# 0
#    5  4  3
#    8  7  6
#    11 10 9
#
DIST_THRESH = np.inf
SCALE = 0.18
GRID = np.array([
    [1.5, 1.5], # orientation
    [1.5, 2.5], # orientation
    [0.5, 2.5], # orientation
    [-1, 1],
    [0, 1],
    [1, 1],
    [-1, 0],
    [0, 0],
    [1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
], dtype=np.float64) * SCALE

ORIENTATION_IDXES = np.array([0, 1, 2])
INV_ORIENTATION_IDXES = np.setdiff1d(np.arange(GRID.shape[0]), ORIENTATION_IDXES)
GRID_WEIGHTS = np.ones(GRID.shape[0])
GRID_WEIGHTS[ORIENTATION_IDXES] = 1.5
BOUNDS = [[-SCALE * 3, SCALE * 3], [-SCALE * 3, SCALE * 3], [0.9, 1.1], [0.9, 1.1], [0, 2 * np.pi]]

MAX_VAL = 1 << 12

# Used to convert read binary into actual index / letter pairing
LETTER_VALUES = np.array(
    [0, 0, 0, 32, 16, 8, 4, 2, 1, 0, 0, 0]
)

IDX_VALUES = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1]
)

def readout(bstring):
    return (np.sum(bstring * LETTER_VALUES) << 3) + np.sum(bstring * IDX_VALUES)

def to_string(val):
    idx = val & 0b111
    letter = val >> 3
    return f"{letter}/{chr(letter + ord('A') - 1)},{idx}"

def true_read(tag):
    c = tag[0]

    # 14, 0
    if c == 'N':
        return np.array([1,1,1,0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    # 19, 1
    elif c == 'S':
        return np.array([1,1,1,0, 1, 0, 0, 1, 1, 0, 0, 1], dtype=int)
    # 6, 2
    elif c == 'F':
        return np.array([1,1,1,0, 0, 0, 1, 1, 0, 0, 1, 0], dtype=int)

def read_match(read, tag):
    return all(read[3:] == true_read(tag)[3:])

def binnify(group):
    return group[0]

CORRECT_READS = [
    ['N',readout(true_read('N'))],
    ['S',readout(true_read('S'))],
    ['F',readout(true_read('F'))],
]
