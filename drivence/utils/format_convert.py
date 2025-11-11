import numpy
import open3d as o3d
import numpy as np


def pc_numpy_2_pcd(xyz: numpy.ndarray) -> o3d.geometry.PointCloud:
    if isinstance(xyz, np.ndarray):
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(xyz)
        return pcd_bg
    return xyz


def pcd_2_pc_numpy(pcd_obj: o3d.geometry.PointCloud) -> numpy.ndarray:
    if isinstance(pcd_obj, o3d.geometry.PointCloud):
        return np.asarray(pcd_obj.points)
    return pcd_obj


def numpy_to_torch(x):
    import torch
    """Convert numpy array to torch tensor if applicable."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x


def torch_to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return x


def list_to_numpy(x):
    if isinstance(x, list):
        return np.array(x)
    return x


def to_numpy(x):
    x = list_to_numpy(x)
    x = torch_to_numpy(x)
    return x


def to_torch(x):
    x = list_to_numpy(x)
    x = numpy_to_torch(x)
    return x


import pandas as pd


def get_list_max_index(list_: list, n: int) -> list:
    """Get indices of top n maximum values in descending order."""
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
    return list(N_large.index)[:n]
