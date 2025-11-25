from typing import List

from astropy import coordinates
from scipy.spatial.transform import Rotation as RR
import numpy as np
import math

def is_right_hand_system(x, y, z):
    """
    判断三个基向量是否构成右手坐标系（Right-Handed Coordinate System）。

    核心逻辑：
    右手坐标系的定义为：x 轴叉乘 y 轴的结果与 z 轴方向一致（允许微小数值误差）。
    通过计算 x×y 的叉乘结果，与 z 轴向量进行近似相等校验。

    Args:
        x (np.ndarray): x 轴基向量，形状为 (2,) 或 (3,)（需与 y、z 维度一致）；
        y (np.ndarray): y 轴基向量，形状为 (2,) 或 (3,)；
        z (np.ndarray): z 轴基向量，形状为 (2,) 或 (3,)。

    Returns:
        bool: 若构成右手坐标系则返回 True，否则返回 False。
    """
    # Cross product of x and y
    cross_product = np.cross(x, y)

    # Check whether the cross product matches z
    return np.allclose(cross_product, z)




def caluate_rotation_matrix(coord_a, coord_b):
    """
    计算两个坐标系之间的旋转信息：2D 场景返回旋转角（弧度），3D 场景返回旋转矩阵对象。

    核心逻辑：
    1. 先校验两个坐标系的左右手性是否一致，不一致则抛出异常；
    2. 计算 A→B 的变换矩阵；
    3. 2D 场景：从变换矩阵提取旋转角（arctan2(matrix[1,0], matrix[0,0])）；
    4. 3D 场景：将变换矩阵包装为 scipy Rotation 对象（支持后续欧拉角、四元数转换）。

    Args:
        coord_a (dict): 源坐标系 A 的基向量，格式为 {"x": np.ndarray, "y": np.ndarray}（2D）或
            {"x": np.ndarray, "y": np.ndarray, "z": np.ndarray}（3D）；
        coord_b (dict): 目标坐标系 B 的基向量，格式与 coord_a 一致。

    Returns:
        Union[float, RR]:
            - 2D 场景：旋转角（弧度），范围 [-π, π]；
            - 3D 场景：scipy Rotation 对象（包含旋转矩阵信息）。
    """
    flag1 = is_right_hand_system(coord_a["x"],coord_a["y"],np.array([0,0,1]))
    flag2 = is_right_hand_system(coord_b["x"],coord_b["y"],np.array([0,0,1]))
    if flag1 != flag2:
        raise ValueError("the two coordinate system is not in the same hand system")
    else:
        trans_matrix = compute_trans_matrix_a2b(coord_a,coord_b)
        result = None
        if len(coord_a["x"]) == 2:
            rotation_angle = np.arctan2(trans_matrix[1,0],trans_matrix[0,0])
            result = rotation_angle
        elif len(coord_a["x"]) == 3:
            rotation_matrix = RR.from_matrix(trans_matrix)
            result = rotation_matrix
        return result

def compute_trans_matrix_a2b(coord_a, coord_b):
    """
    计算从坐标系 A 到坐标系 B 的 2×2 变换矩阵（基于基向量映射）。

    核心逻辑：
    1. 坐标系的基向量构成矩阵（每列为一个基向量）；
    2. 变换矩阵 = B 的基矩阵 × A 的基矩阵的逆（满足 B = A × T，即 T = A⁻¹ × B）。

    Args:
        coord_a (dict): 源坐标系 A 的基向量，格式为 {"x": np.ndarray, "y": np.ndarray}，
            基向量需为 2D（形状 (2,)）；
        coord_b (dict): 目标坐标系 B 的基向量，格式与 coord_a 一致。

    Returns:
        np.ndarray: 从 A 到 B 的变换矩阵，形状为 (2, 2)。
    """
    # Build basis matrices (each column is a basis vector)
    matrix_a = np.array([coord_a["x"], coord_a["y"]]).T
    matrix_b = np.array([coord_b["x"], coord_b["y"]]).T

    # Transformation matrix
    transformation_matrix = matrix_b @ np.linalg.inv(matrix_a)

    return transformation_matrix

def compute_trans_matrix_b2a_by_a2b(T_a2b):
    """
    通过 A→B 的变换矩阵，计算 B→A 的逆变换矩阵（利用矩阵逆的性质）。

    Args:
        T_a2b (np.ndarray): A→B 的变换矩阵，形状为 (N, N)（N=2 或 3）。

    Returns:
        np.ndarray: B→A 的逆变换矩阵，形状与 T_a2b 一致。
    """
    T_b2a = np.linalg.inv(T_a2b)
    return T_b2a

def transform_point_a2b(point_a,T_a2b):
    """
    将点从坐标系 A 变换到坐标系 B（基于变换矩阵 T_A→B）。

    核心逻辑：
    - 1D 点（单个点）：直接用矩阵乘向量（T × point）；
    - 2D 点（批量点）：点矩阵每行一个点，用点矩阵乘变换矩阵的转置（point × Tᵀ）。

    Args:
        point_a (np.ndarray): 坐标系 A 中的点，形状为 (N,)（1D 单个点）或 (M, N)（2D 批量点），
            N 为维度（2 或 3），M 为点数量；
        T_a2b (np.ndarray): A→B 的变换矩阵，形状为 (N, N)。

    Returns:
        np.ndarray: 坐标系 B 中的点，形状与 point_a 一致。
    """
    if point_a.ndim == 1:
        return T_a2b @ point_a
    elif point_a.ndim == 2:
        return point_a @ T_a2b.T
    else:
        raise ValueError(f"point_a {point_a.shape} must be a 1D or 2D array, transform matrix shape is {T_a2b.shape}")


def get_euler_from_rotate_matrix(R: np.ndarray) -> List[float]:
    """
    从 3D 旋转矩阵中提取欧拉角（默认 XYZ 顺序，弧度制）。

    Args:
        R (np.ndarray): 3D 旋转矩阵，形状为 (3, 3)（必须是合法的正交矩阵）；
        euler_type (str, 可选): 欧拉角旋转顺序，默认 "XYZ"（其他支持如 "ZYX"、"YXZ" 等）。

    Returns:
        List[float]: 欧拉角列表，顺序与 euler_type 一致，单位为弧度。
    """
    euler_type = "XYZ"
    sciangle = RR.from_matrix(R).as_euler(euler_type)
    return [float(angle) for angle in sciangle]


def spherical_to_cartesian(phi, theta, r, input_type="degree",origin=None):
    """
    将球坐标转换为笛卡尔坐标（XYZ），支持自定义原点偏移。

    球坐标定义（遵循 astropy 标准）：
    - phi (方位角)：在 XY 平面内，从 X 轴正方向逆时针旋转的角度；
    - theta (极角)：从 Z 轴正方向向下旋转的角度；
    - r (半径)：点到原点的距离。

    Args:
        phi (float): 方位角（phi）；
        theta (float): 极角（theta）；
        r (float): 半径（r）；
        input_type (str, 可选): 输入角度单位，默认 "degree"（度），可选 "radian"（弧度）；
        origin (Optional[Union[List, Tuple, np.ndarray]], 可选): 笛卡尔坐标原点偏移，
            形状为 (3,)，默认 None（无偏移，原点为 (0,0,0)）。

    Returns:
        Tuple[float, float, float]: 笛卡尔坐标 (x, y, z)。
    """
    if input_type == "degree":
        phi = math.radians(phi)
        theta = math.radians(theta)
    x, y, z = coordinates.spherical_to_cartesian(r, phi, theta)
    x, y, z = x.value, y.value, z.value
    if origin is not None:
        x = x + origin[0]
        y = y + origin[1]
        z = z + origin[2]
    return x, y, z


# return phi theta
def cartesian_to_spherical(x, y, z, to_degree=True, return_r=False,origin=None):
    """
    将笛卡尔坐标（XYZ）转换为球坐标，支持自定义原点偏移和批量处理。

    球坐标定义（与 spherical_to_cartesian 对应）：
    - 输出第一个值（latitude）：方位角 phi；
    - 输出第二个值（longitude）：极角 theta；
    - 输出第三个值（r）：半径（可选返回）。

    Args:
        x (Union[float, np.ndarray]): 笛卡尔坐标 x 分量（支持单个值或批量数组）；
        y (Union[float, np.ndarray]): 笛卡尔坐标 y 分量；
        z (Union[float, np.ndarray]): 笛卡尔坐标 z 分量；
        to_degree (bool, 可选): 输出角度单位是否为度，默认 True（度），False 为弧度；
        return_r (bool, 可选): 是否返回半径 r，默认 False；
        origin (Optional[Union[List, Tuple, np.ndarray]], 可选): 笛卡尔坐标原点偏移，
            形状为 (3,)，默认 None（无偏移）。

    Returns:
        Union[Tuple[float, float], Tuple[float, float, float], Tuple[np.ndarray, ...]]:
            - 若 return_r=False：(phi, theta)；
            - 若 return_r=True：(phi, theta, r)；
            - 输入为数组时，输出为对应形状的 numpy 数组。
    """
    if origin is not None:
        x = x - origin[0]
        y = y - origin[1]
        z = z - origin[2]
    r, latitude, longitude = coordinates.cartesian_to_spherical(x, y, z)  #
    latitude, longitude, r = latitude.value, longitude.value, r.value
    if to_degree:
        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)
    if return_r:
        return latitude, longitude, r
    return latitude, longitude
