import os
import pickle
import numpy as np
import trimesh
from itertools import product
from scipy.spatial import KDTree

def points_normalize(pc, factor, offset):
    bmin, bmax = np.min(pc, axis=0), np.max(pc, axis=0)
    center, scale = (bmin + bmax) / 2., np.max(bmax - bmin) / 2.
    pc -= center
    pc *= factor / scale
    pc += offset

    return pc

def precompute_index(pc, k, cube_shift=True):
    step = 2. / k
    pcid = np.floor((pc + 1) / step).astype(int)
    grid = np.zeros((k, k, k), dtype=bool)
    grid[pcid[:, 0], pcid[:, 1], pcid[:, 2]] = True

    if cube_shift:
        # shift to neighbor(26/27)
        ijl = product(range(3), range(3), range(3))
        tmp_mask = np.copy(grid[1:-1, 1:-1, 1:-1])
        for i, j, l in ijl:
            grid[i:k - 2 + i, j:k - 2 + j, l:k - 2 + l] = np.logical_or(tmp_mask, grid[i:k - 2 + i, j:k - 2 + j, l:k - 2 + l])

    cid = np.argwhere(grid)
    return pcid, cid

def KNN_idx(pc, leafsize):
    tree = KDTree(pc, leafsize=leafsize)
    _, idx = tree.query(pc, k=leafsize)
    return idx

def output_pc_data(pc, k, output_path, offset=None, leaf_size=8):
    pcid, cid = precompute_index(pc, k)
    pc_KNN_idx = KNN_idx(pc, leafsize=leaf_size)

    res = {
        'pc': pc,
        'pc_grid_idx': pcid,
        'pc_KNN_idx': pc_KNN_idx,
        'cube_grid_idx': cid,
        'grid_size': k,
        'leaf_size': leaf_size,
        'stable_offset': offset
    }

    with open(output_path, 'wb') as f:
        pickle.dump(res, f)

def convert_single_obj(obj_path, output_path, k=64, leaf_size=8, ub_numpc=50000):
    mesh = trimesh.load(obj_path, process=False)
    num_pc = mesh.vertices.shape[0]

    if num_pc <= ub_numpc:
        pc = np.asarray(mesh.vertices)
    else:
        pc = mesh.sample(ub_numpc)

    offset = 0  # Set offset to 0
    pc = points_normalize(pc, factor=0.9, offset=offset)

    output_pc_data(pc, k, output_path, offset, leaf_size=leaf_size)

if __name__ == '__main__':
    # Specify your obj file and output path
    obj_path = '/home/chris/Code/PointClouds/data/FLIPscans/GrateAndCover/CoverOBJ.obj'
    output_path = '/home/chris/Code/NerVE/demo/0002/pc_obj.pkl'

    # Call the function to convert a single obj file
    convert_single_obj(obj_path, output_path)