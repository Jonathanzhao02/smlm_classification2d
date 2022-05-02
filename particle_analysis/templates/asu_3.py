import numpy as np

# TEMPLATE 2: 6x8 grid
DIST_THRESH = np.inf
SCALE = 0.08
GRID = np.zeros((48,2), dtype=np.float64)

# 0  1  2  3  4  5  6  7
# 8  9  10 11 12 13 14 15
# 16 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30 31
# 32 33 34 35 36 37 38 39
# 40 41 42 43 44 45 46 47
# Keep in mind origami is FLIPPED VERTICALLY from image -> matplotlib

for i in range(6):
    for j in range(8):
        GRID[i * 8 + j] = [-3.5 + j, 2.5 - i]

GRID *= SCALE
ORIENTATION_IDXES = np.array([0, 1, 8, 6, 7, 15, 32, 40, 41])
INV_ORIENTATION_IDXES = np.setdiff1d(np.arange(GRID.shape[0]), ORIENTATION_IDXES)
GRID_WEIGHTS = np.ones(GRID.shape[0])
GRID_WEIGHTS[ORIENTATION_IDXES] = 1.5
BOUNDS = [[-SCALE * 3, SCALE * 3], [-SCALE * 3, SCALE * 3], [0.9, 1.1], [0.9, 1.1], [0, 2 * np.pi]]

# Used to convert read binary into actual index / letter pairing
LETTER_VALUES = np.array(
    [1, 2, 4, 8, 16, 32, 0, 0, 0, 0, 0, 0]
)

IDX_VALUES = np.array(
    [0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32]
)

# 3-repetition
REPETITION_PAIRS = np.array([
    [14, 22, 23], [5, 13, 21], [4, 12, 20], [3, 11, 19], [2, 10, 18], [9, 16, 17],
    [30, 31, 38], [29, 37, 45], [28, 36, 44], [27, 35, 43], [26, 34, 42], [24, 25, 33],
], dtype=int)

def apply_repetition(raw):
    if raw is None:
        return None

    read = np.zeros(REPETITION_PAIRS.shape[0], dtype=int)

    for i,pair in enumerate(REPETITION_PAIRS):
        read[i] = any(raw[pair])
    
    return read

def inv_apply_repetition(read):
    raw = np.zeros(GRID.shape[0], dtype=int)

    for i,idx in enumerate(read):
        raw[REPETITION_PAIRS[i]] = idx

    raw[ORIENTATION_IDXES] = 1
    return raw

def true_read(tag):
    c = tag[0]

    # 1, 1
    if c == 'A':
        return inv_apply_repetition([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    # 19, 2
    elif c == 'S':
        return inv_apply_repetition([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0])
    # 21, 3
    elif c == 'U':
        return inv_apply_repetition([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1])

def readout(bstring):
    bstring = apply_repetition(bstring)
    return (np.sum(bstring * LETTER_VALUES) << 6) + np.sum(bstring * IDX_VALUES)

def to_string(val):
    idx = val & 0b111111
    letter = val >> 6
    return f"{letter}/{chr(letter + ord('A') - 1)},{idx}"

def read_match(read, tag):
    read = apply_repetition(read)
    return all(read == apply_repetition(true_read(tag)))

def binnify(group):
    return group[0]
