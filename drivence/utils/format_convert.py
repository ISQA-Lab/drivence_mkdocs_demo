import numpy
import open3d as o3d
import numpy as np


def pc_numpy_2_pcd(xyz: numpy.ndarray) -> o3d.geometry.PointCloud:
    """
    将numpy格式的3D点云转换为Open3D的PointCloud对象。

    Args:
        xyz (np.ndarray): 3D点云数组，形状为 (N, 3)，每行对应 [x, y, z] 坐标。

    Returns:
        o3d.geometry.PointCloud: Open3D点云对象；若输入非numpy数组，直接返回原输入。
    """
    if isinstance(xyz, np.ndarray):
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(xyz)
        return pcd_bg
    return xyz


def pcd_2_pc_numpy(pcd_obj: o3d.geometry.PointCloud) -> numpy.ndarray:
    """
    将Open3D的PointCloud对象转换为numpy格式的3D点云数组。

    Args:
        pcd_obj (o3d.geometry.PointCloud): Open3D点云对象。

    Returns:
        np.ndarray: 3D点云数组，形状为 (N, 3)；若输入非Open3D点云对象，直接返回原输入。
    """
    if isinstance(pcd_obj, o3d.geometry.PointCloud):
        return np.asarray(pcd_obj.points)
    return pcd_obj


def numpy_to_torch(x):
    """
    将numpy数组转换为torch张量（float32类型）。

    Args:
        x (Union[np.ndarray, Any]): 输入数据（支持numpy数组或其他类型）。

    Returns:
        Union[torch.Tensor, Any]: torch张量（float32）；若输入非numpy数组，直接返回原输入。
    """
    import torch
    """Convert numpy array to torch tensor if applicable."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x


def torch_to_numpy(x):
    """
    将torch张量转换为numpy数组。

    Args:
        x (Union[torch.Tensor, Any]): 输入数据（支持torch张量或其他类型）。

    Returns:
        Union[np.ndarray, Any]: numpy数组；若输入非torch张量，直接返回原输入。
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return x


def list_to_numpy(x):
    """
    将列表转换为numpy数组。

    Args:
        x (Union[List[Any], Any]): 输入数据（支持列表或其他类型）。

    Returns:
        Union[np.ndarray, Any]: numpy数组；若输入非列表，直接返回原输入。
    """
    if isinstance(x, list):
        return np.array(x)
    return x


def to_numpy(x):
    """
    统一将输入数据转换为numpy数组（支持列表、torch张量、numpy数组）。

    转换优先级：列表 → numpy数组 → torch张量 → numpy数组。

    Args:
        x (Union[List[Any], torch.Tensor, np.ndarray, Any]): 输入数据（支持多种类型）。

    Returns:
        np.ndarray: 转换后的numpy数组。
    """
    x = list_to_numpy(x)
    x = torch_to_numpy(x)
    return x


def to_torch(x):
    """
    统一将输入数据转换为torch张量（float32类型，支持列表、numpy数组、torch张量）。

    转换优先级：列表 → numpy数组 → torch张量（float32）。

    Args:
        x (Union[List[Any], np.ndarray, torch.Tensor, Any]): 输入数据（支持多种类型）。

    Returns:
        torch.Tensor: 转换后的torch张量（float32）。
    """
    x = list_to_numpy(x)
    x = numpy_to_torch(x)
    return x


import pandas as pd


def get_list_max_index(list_: list, n: int) -> list:
    """
    获取列表中前n个最大值的索引，按数值降序排列。

    核心逻辑：使用pandas DataFrame排序，提取前n个最大数值对应的原始索引。

    Args:
        list_ (List[float]): 输入数值列表（需为可排序的数值类型）；
        n (int): 需获取的最大数值的个数（若n≥列表长度，返回所有元素的索引，按降序排列）。

    Returns:
        List[int]: 前n个最大值的索引列表，顺序为数值从大到小对应的原始索引。
    """
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
    return list(N_large.index)[:n]
