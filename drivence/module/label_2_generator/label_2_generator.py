"""
Label 2 Generator for SemanticKITTI
Generates KITTI-format label_2 files from semantic labels for collision detection.
"""

import os
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, Optional


SEMANTIC_TO_NAME = {
    # static
    10: "Car",
    11: "Bicycle",
    13: "Bus",
    15: "Motorcycle",
    18: "Truck",
    30: "Person",
    31: "Bicyclist",
    32: "Motorcyclist",
    # moving counterparts
    252: "Car",
    257: "Bus",
    258: "Truck",
    253: "Bicyclist",
    254: "Person",
    255: "Motorcyclist",
    259: "Car",  # treat moving-other-vehicle as a generic car surrogate
}


def read_velodyne_bin(path: str) -> np.ndarray:
    """
    读取 Velodyne 激光雷达的二进制（.bin）文件，提取 3D 点云坐标（x, y, z）。

    Velodyne 激光雷达输出的 .bin 文件采用固定格式存储：每个点云数据包含 4 个 float32 类型值，
    依次为 x（x轴坐标）、y（y轴坐标）、z（z轴坐标）、intensity（反射强度）。
    该函数读取文件后解析数据格式，仅提取前 3 个维度的空间坐标，返回标准化的点云数组，
    适用于点云可视化、特征提取、跨模态融合等后续处理场景。

    Args:
        path (str): Velodyne .bin 文件的绝对路径或相对路径（含文件名）。
            示例："/data/velodyne/000001.bin"、"../dataset/frame_001.bin"。

    Returns:
        np.ndarray: 3D 点云坐标数组，形状为 (N, 3)，数据类型为 np.float32。
            - N: 点云中点的数量（由文件大小决定，每个点占 16 字节：4 个 float32）；
            - 3: 每个点的 x、y、z 三维空间坐标，坐标单位与激光雷达输出一致（通常为米）。
    """
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Invalid velodyne bin file: {path}")
    pts = data.reshape((-1, 4))[:, :3]
    return pts


def read_semantic_label(path: str) -> np.ndarray:
    """
    读取语义标签文件（支持 .label 二进制格式和 .npy numpy 格式），返回标准化的一维语义标签数组。

    针对激光雷达点云/图像语义分割任务的标签读取需求，适配两种主流标签存储格式：
    1. .label 格式：Velodyne 点云语义分割任务常用二进制格式，每个标签为 32 位无符号整数，低 16 位存储语义类别 ID，高 16 位通常存储实例 ID；
    2. .npy 格式：numpy 序列化格式，直接存储语义标签数组，支持快速加载。
    函数自动识别文件后缀并选择对应解析逻辑，最终返回一维整数型语义标签数组，适配后续标签处理、可视化或模型训练场景。

    Args:
        path (str): 语义标签文件的绝对路径或相对路径（含文件名）。
            支持后缀：.label（二进制格式）、.npy（numpy 格式）；
            示例："/data/semantic/000001.label"、"../dataset/frame_001_semantic.npy"。

    Returns:
        np.ndarray: 一维语义标签数组，形状为 (N,)，数据类型为 np.int32。
            - N: 标签数量（与对应点云/图像的像素/点数量一致）；
            - 数组元素：语义类别 ID（整数），取值范围由数据集的标签体系决定（如 0 表示背景、1 表示车辆等）。
    """
    if path.endswith(".label"):
        labels = np.fromfile(path, dtype=np.uint32)
        semantic = labels & 0xFFFF
        return semantic.astype(np.int32)
    elif path.endswith(".npy"):
        return np.load(path).astype(np.int32).reshape(-1)
    else:
        raise ValueError(f"Unsupported semantic label file: {path}")


def cluster_points_dbscan(points: np.ndarray, eps: float, min_samples: int = 5) -> list:
    """
      基于 DBSCAN（密度聚类算法）对 3D 点云在 XY 平面进行聚类，提取有效目标点云簇。

      针对激光雷达点云的目标检测场景，忽略 Z 轴高度信息，仅基于 XY 平面的水平分布进行密度聚类，
      筛选出满足最小点数要求的聚类簇（视为有效目标），过滤噪声点和过小的离散点集。
      适用于车辆、行人等地面目标的初步分割，为后续目标识别、边界框拟合提供基础。

      核心逻辑：
      1. 数据预处理：若输入点云为空，直接返回空列表；否则提取点云的 XY 平面坐标（忽略 Z 轴）；
      2. DBSCAN 聚类：使用指定的邻域半径（eps）和最小点数（min_samples）执行密度聚类；
      3. 结果筛选：过滤噪声点（标签为 -1）和点数小于 5 的无效簇，保留有效目标点云簇；
      4. 结果返回：返回包含所有有效簇的列表，每个簇为原始 3D 点云的子集（保留 XYZ 坐标）。

      Args:
          points (np.ndarray): 输入 3D 点云数组，形状为 (N, 3)，数据类型为数值型。
              - N: 点云中点的数量；
              - 3: 每个点的 X、Y、Z 三维坐标（单位通常为米）。
          eps (float): DBSCAN 算法的邻域半径（单位与坐标一致，通常为米），
              用于定义“密度”的邻域范围，需根据点云密度调整（如点云密集时取较小值，稀疏时取较大值）。
          min_samples (int, 可选): 形成有效聚类簇的最小点数阈值，默认 5。
              表示一个簇中至少包含的点数，用于过滤噪声和离散小簇。

      Returns:
          list[np.ndarray]: 有效目标点云簇列表，每个元素为一个 np.ndarray，形状为 (M, 3)：
              - M: 该簇中点的数量（M ≥ 5，即满足最小点数要求）；
              - 3: 每个点保留原始 XYZ 三维坐标；
              若输入点云为空或无有效簇，返回空列表。
      """
    if len(points) == 0:
        return []
    xy = points[:, :2]  # Use only X, Y for clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = clustering.labels_
    clusters = []
    for label in np.unique(labels):
        if label == -1:  # Skip noise points
            continue
        cluster_points = points[labels == label]
        if len(cluster_points) >= 5:  # At least 5 points to be considered a valid object
            clusters.append(cluster_points)
    return clusters


def write_label2(path: str, boxes: list):
    """
    写入 KITTI 格式的 label_2 标注文件，存储 3D 目标边界框及类别信息，适配 KITTI 目标检测数据集标准。

    KITTI label_2 文件为纯文本格式，每行对应一个目标的标注信息，字段顺序和格式严格遵循 KITTI 数据集规范，
    支持后续使用 KITTI 官方工具或主流目标检测框架（如 PointPillars、SECOND）进行数据加载、训练或评估。
    函数自动创建输出路径的父目录（若不存在），确保文件正常写入。

    核心规范（KITTI label_2 字段说明，共 15 个字段）：
    1. 目标类别（name）          2. 截断程度（truncated）    3. 遮挡程度（occluded）
    4. 观测角度（alpha）         5. 2D 边界框左（x1）       6. 2D 边界框上（y1）
    7. 2D 边界框右（x2）         8. 2D 边界框下（y2）       9. 3D 边界框高度（h）
    10. 3D 边界框宽度（w）       11. 3D 边界框长度（l）     12. 3D 边界框中心 x（x）
    13. 3D 边界框中心 y（y）     14. 3D 边界框中心 z（z）   15. 旋转角（ry）
    本函数中，字段 2-8（截断、遮挡、观测角度、2D 边界框）统一填充默认值（-1/-1/0/0/0/0/0），仅需输入核心的 3D 信息。

    Args:
        path (str): label_2 标注文件的输出路径（含文件名），格式为 "xxx/xxx.txt"。
            示例："/data/kitti/training/label_2/000001.txt"、"../dataset/frame_001_label.txt"。
        boxes (list[tuple]): 3D 目标边界框列表，每个元素为一个元组，包含 8 个字段（对应 KITTI 核心标注信息）：
            - name (str): 目标类别名称（需符合 KITTI 类别规范，如 "Car"、"Pedestrian"、"Cyclist" 等）；
            - h (float): 3D 边界框高度（单位：米）；
            - w (float): 3D 边界框宽度（单位：米）；
            - l (float): 3D 边界框长度（单位：米）；
            - x (float): 3D 边界框中心 x 坐标（单位：米，遵循 KITTI 坐标系）；
            - y (float): 3D 边界框中心 y 坐标（单位：米，遵循 KITTI 坐标系）；
            - z (float): 3D 边界框中心 z 坐标（单位：米，遵循 KITTI 坐标系）；
            - ry (float): 目标绕 Y 轴的旋转角（单位：弧度，遵循 KITTI 旋转角定义）。

    Returns:
        None: 函数无显式返回值，标注信息直接写入指定路径的文件中。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name, h, w, l, x, y, z, ry in boxes:
            f.write(f"{name} -1 -1 0 0 0 0 0 {h:.3f} {w:.3f} {l:.3f} {x:.3f} {y:.3f} {z:.3f} {ry:.3f}\n")


def read_kitti_calib(calib_txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 KITTI 数据集的标定文件（.txt），提取并返回相机矫正矩阵 R0 和 Velodyne 到相机的变换矩阵 Tr。

    KITTI 标定文件存储了多传感器（激光雷达、相机、雷达）之间的外参和内参，本函数专注于提取激光雷达与相机融合所需的核心矩阵：
    1. R0（相机矫正矩阵）：用于矫正相机的径向畸变，将相机坐标系下的点转换为无畸变的归一化平面坐标；
    2. Tr（Velodyne→Camera 变换矩阵）：用于将激光雷达坐标系下的 3D 点云转换到相机坐标系，是跨模态（点云-图像）融合的关键外参。
    函数兼容 KITTI 标定文件中矩阵的多种常见键名（如 R0_rect/R_rect、Tr_velo_to_cam/Tr_velo2cam），提升鲁棒性。

    核心逻辑：
    1. 文件读取：按行解析标定文件，过滤无效行（不含 ':' 的行）；
    2. 键值提取：拆分每行的键名（key）和数值字符串（rest），将数值字符串转换为浮点数列表；
    3. 矩阵解析：
       - 若键名匹配相机矫正矩阵（R0_rect/R_rect 等），取前 9 个数值reshape为 (3, 3) 矩阵；
       - 若键名匹配激光雷达到相机变换矩阵（Tr_velo_to_cam 等），取前 12 个数值reshape为 (3, 4) 矩阵；
    4. 容错处理：Tr 矩阵为必需项（缺失则抛出异常），R0 矩阵缺失时默认使用 3x3 单位矩阵。

    Args:
        calib_txt_path (str): KITTI 标定文件的绝对路径或相对路径（含文件名）。
            示例："/data/kitti/training/calib/000001.txt"、"../dataset/frame_001_calib.txt"。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 包含两个矩阵的元组，顺序如下：
            1. R0 (np.ndarray): 相机矫正矩阵，形状为 (3, 3)，数据类型为 np.float32。
                - 作用：矫正相机畸变，将相机坐标系点转换为无畸变坐标；
                - 若文件中未找到对应键名，返回 3x3 单位矩阵（np.eye(3)）。
            2. Tr (np.ndarray): Velodyne 激光雷达到相机的变换矩阵，形状为 (3, 4)，数据类型为 np.float32。
                - 作用：实现点云坐标从激光雷达坐标系 → 相机坐标系的转换（公式：P_cam = R0 @ (Tr @ [P_velo; 1])）；
                - 矩阵格式：前 3 列为旋转矩阵，第 4 列为平移向量，即 [R | t]，其中 R 为 3x3 旋转矩阵，t 为 3x1 平移向量。
    """
    R0 = None
    Tr = None
    with open(calib_txt_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, rest = line.split(':', 1)
            vals = [float(x) for x in rest.strip().split()]
            k = key.strip()
            if k in ('R0_rect', 'R_rect', 'R_rect_00') and len(vals) >= 9:
                R0 = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
            elif k in ('Tr_velo_to_cam', 'Tr_velo2cam', 'Tr') and len(vals) >= 12:
                Tr = np.array(vals[:12], dtype=np.float32).reshape(3, 4)

    if Tr is None:
        raise ValueError(f"Invalid calib file (missing Tr): {calib_txt_path}")
    if R0 is None:
        R0 = np.eye(3, dtype=np.float32)
    return R0, Tr


def lidar_to_rect(points_lidar: np.ndarray, R0: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """
    将激光雷达（LiDAR）坐标系下的 3D 点云转换到矫正后相机（rect camera）坐标系，为跨模态融合提供空间对齐。

    该函数是点云-图像跨模态处理的核心坐标转换接口，通过两步变换实现坐标映射：
    1. 激光雷达 → 相机坐标系：利用外参矩阵 Tr（Velodyne→Camera）完成旋转+平移变换；
    2. 相机 → 矫正后相机坐标系：利用矫正矩阵 R0 消除相机径向畸变，得到归一化无畸变坐标。
    转换后的数据可直接用于点云投影到图像平面、跨模态特征对齐等场景（如 Li-Fusion 中的特征融合前处理）。

    核心变换公式：
    1. 齐次坐标扩展：P_h = [P_lidar; 1] （将 3D 点扩展为 4D 齐次坐标，支持平移变换）；
    2. 激光雷达到相机：P_cam = Tr × P_h^T （矩阵乘法，得到相机坐标系下的 3D 点）；
    3. 相机到矫正后相机：P_rect = R0 × P_cam^T （矩阵乘法，消除畸变，得到最终矫正坐标）。

    Args:
        points_lidar (np.ndarray): 激光雷达坐标系下的 3D 点云数组，形状为 (N, 3)，数据类型需为数值型。
            - N: 点云中点的数量；
            - 3: 每个点的 X、Y、Z 三维坐标（单位：米，遵循 LiDAR 坐标系定义）；
            若输入为其他数据类型（如 int），内部会自动转换为 np.float32。
        R0 (np.ndarray): 相机矫正矩阵，形状为 (3, 3)，数据类型为 np.float32。
            - 作用：消除相机径向畸变，将原始相机坐标系点转换为无畸变的矫正坐标系点；
            通常从 KITTI 标定文件中读取（如 read_kitti_calib 函数输出），缺失时可用单位矩阵。
        Tr (np.ndarray): 激光雷达到相机的外参变换矩阵，形状为 (3, 4)，数据类型为 np.float32。
            - 格式：前 3 列为旋转矩阵（R），第 4 列为平移向量（t），即 [R | t]；
            - 作用：实现激光雷达坐标系到相机坐标系的旋转+平移变换，需与点云、相机的坐标系严格匹配。

    Returns:
        np.ndarray: 矫正后相机坐标系下的 3D 点云数组，形状为 (N, 3)，数据类型为 np.float32。
            - N: 点数量与输入保持一致；
            - 3: 每个点的 X_rect、Y_rect、Z_rect 三维矫正坐标（单位：米），无畸变，可直接用于后续投影计算。
    """
    ones = np.ones((points_lidar.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([points_lidar.astype(np.float32), ones], axis=1)  # (N,4)
    pts_cam = (Tr @ pts_h.T).T  # (N,3)
    pts_rect = (R0 @ pts_cam.T).T  # (N,3)
    return pts_rect


def rect_to_lidar(points_rect: np.ndarray, R0: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """
    将矫正后相机（rect camera）坐标系下的 3D 点转换到激光雷达（LiDAR）坐标系，实现跨模态坐标反向映射。

    该函数是 `lidar_to_rect` 的逆操作，为点云-图像跨模态处理的反向对齐提供支持（如从图像标注反推点云区域、跨模态结果验证）。
    通过两步逆变换还原激光雷达坐标：
    1. 矫正后相机 → 相机坐标系：利用矫正矩阵 R0 的转置（因 R0 为正交矩阵，逆矩阵=转置）消除矫正效果；
    2. 相机 → 激光雷达坐标系：利用 Tr 矩阵的逆变换（旋转矩阵转置+平移向量反向）还原原始激光雷达坐标。
    适用于图像引导的点云分割、标注转换、跨模态结果对齐等场景。

    核心逆变换公式（基于 `lidar_to_rect` 正向变换推导）：
    1. 矫正后相机到相机：P_cam = R0^T × P_rect^T （R0 为正交矩阵，逆矩阵=转置，消除矫正）；
    2. 相机到激光雷达：P_lidar = R^T × (P_cam - t) （R 为 Tr 中的旋转矩阵，t 为 Tr 中的平移向量，反向实现旋转+平移逆操作）。

    Args:
        points_rect (np.ndarray): 矫正后相机坐标系下的 3D 点数组，形状为 (N, 3) 或 (3,)，数据类型需为数值型。
            - N: 点的数量（N≥1）；若为单一点，支持 (3,) 一维格式；
            - 3: 每个点的 X_rect、Y_rect、Z_rect 三维坐标（单位：米，无畸变）；
            输入为一维数组时，内部会自动reshape为 (1, 3) 处理，输出保持一维格式。
        R0 (np.ndarray): 相机矫正矩阵，形状为 (3, 3)，数据类型为 np.float32。
            - 需与正向变换（lidar_to_rect）使用的 R0 完全一致，确保逆变换准确性；
            通常从 KITTI 标定文件读取（如 read_kitti_calib 函数输出），为正交矩阵（R0^T = R0^-1）。
        Tr (np.ndarray): 激光雷达到相机的外参变换矩阵，形状为 (3, 4)，数据类型为 np.float32。
            - 格式：前 3 列为旋转矩阵 R（3x3），第 4 列为平移向量 t（3x1），即 [R | t]；
            需与正向变换使用的 Tr 完全一致，R 为正交矩阵（R^T = R^-1）。

    Returns:
        np.ndarray: 激光雷达坐标系下的 3D 点数组，形状与输入 points_rect 对应：
            - 若输入为 (N, 3)，输出为 (N, 3)；
            - 若输入为 (3,)（单一点），输出为 (3,)；
            数据类型为 np.float32，每个点的 X、Y、Z 坐标（单位：米，遵循 LiDAR 坐标系定义）。
    """
    # Step 1: rect -> cam: cam = R0^T @ rect (R0 is orthogonal)
    if len(points_rect.shape) == 1:
        points_rect = points_rect.reshape(1, -1)
    points_cam = (R0.T @ points_rect.T).T  # (N, 3)

    # Step 2: cam -> velo: need inverse of Tr
    # Tr is 3x4: [R | t], where cam = R @ velo + t
    # So: velo = R^-1 @ (cam - t) = R^T @ (cam - t) (R is rotation, so R^-1 = R^T)
    R = Tr[:, :3]  # 3x3 rotation matrix
    t = Tr[:, 3]  # 3x1 translation vector

    # Apply inverse transformation for each point
    points_lidar = (R.T @ (points_cam - t).T).T  # (N, 3)

    if points_lidar.shape[0] == 1:
        return points_lidar[0]
    return points_lidar


class SemanticKITTILabel2Generator:
    """Generate KITTI-format label_2 files from semantic labels."""
    
    def __init__(self, eps_car: float = 1.2, eps_truck: float = 2.0):
        """
        Initialize label_2 generator.
        
        Args:
            eps_car: DBSCAN eps for cars
            eps_truck: DBSCAN eps for trucks/buses
        """
        self.eps_car = eps_car
        self.eps_truck = eps_truck
        
        # DBSCAN eps map for each semantic ID
        self.dbscan_eps_map = {
            10: eps_car,    # Car
            11: 0.8,        # Bicycle
            13: eps_truck,  # Bus
            15: 0.8,        # Motorcycle
            18: eps_truck,  # Truck
            30: 0.6,        # Person
            31: 0.8,        # Bicyclist
            32: 0.8,        # Motorcyclist
            252: eps_car,
            257: eps_truck,
            258: eps_truck,
            253: 0.8,
            254: 0.6,
            255: 0.8,
            259: eps_car,
        }
    
    def generate_label2_from_semantic(
        self,
        frame_id: int,
        velodyne_path: str,
        semantic_label_path: str,
        calib_path: str,
        output_path: str
    ) -> bool:
        """
        基于激光雷达点云的语义标签，自动生成 KITTI 格式的 label_2 3D 目标检测标注文件。

        核心流程为“语义筛选→密度聚类→坐标转换→边界框计算→标注写入”，无需人工标注，
        可批量将语义分割结果转换为目标检测标注，适配 KITTI 数据集格式要求，支持后续模型训练或评估。
        适配车辆、行人等多类目标，通过 DBSCAN 聚类分割同一语义下的不同实例，基于聚类点云计算边界框维度与位置。

        核心逻辑：
        1. 数据读取：加载激光雷达点云、语义标签、传感器标定参数；
        2. 数据校验：确保点云与语义标签数量匹配，避免维度不兼容；
        3. 语义筛选：按语义ID筛选出各类目标对应的点云子集；
        4. 实例聚类：对每类目标点云用 DBSCAN 聚类，分割不同实例（如多辆汽车）；
        5. 坐标转换：将聚类点云从激光雷达坐标系转换到矫正后相机坐标系（适配 KITTI 标注规范）；
        6. 边界框计算：基于聚类点云的极值计算边界框维度（高/宽/长）和中心坐标；
        7. 标注写入：按 KITTI label_2 格式写入标注文件，返回生成结果状态。

        Args:
            frame_id (int): 帧ID，用于日志输出和错误定位（如“frame_0001”的ID为1）。
            velodyne_path (str): 激光雷达点云二进制文件（.bin）的路径（含文件名）。
                示例："/data/kitti/velodyne/000001.bin"、"../dataset/frame_001_velo.bin"。
            semantic_label_path (str): 语义标签文件路径（含文件名），支持 .label 或 .npy 格式。
                示例："/data/kitti/semantic/000001.label"、"../dataset/frame_001_sem.npy"。
            calib_path (str): KITTI 传感器标定文件（.txt）的路径（含文件名），用于读取坐标转换所需的 R0 和 Tr 矩阵。
                示例："/data/kitti/calib/000001.txt"、"../dataset/frame_001_calib.txt"。
            output_path (str): 生成的 KITTI label_2 标注文件输出路径（含文件名）。
                示例："/data/kitti/label_2/000001.txt"、"../output/frame_001_label.txt"。

        Returns:
            bool: 生成结果状态：
                - True：标注文件成功生成并写入指定路径；
                - False：生成过程中出现错误（如文件缺失、维度不匹配），未生成有效标注。
        """
        try:
            # Read point cloud
            pts_lidar = read_velodyne_bin(velodyne_path)

            # Read semantic labels
            if not os.path.exists(semantic_label_path):
                print(f"[WARN] No semantic labels for frame {frame_id}, skip")
                return False

            sem = read_semantic_label(semantic_label_path)

            if sem.shape[0] != pts_lidar.shape[0]:
                print(f"[WARN] Mismatch points/labels for frame {frame_id}, skip")
                return False

            # Read calibration
            R0, Tr = read_kitti_calib(calib_path)

            # Generate boxes
            boxes = []
            for sem_id, name in SEMANTIC_TO_NAME.items():
                mask = sem == sem_id
                pts_c = pts_lidar[mask]
                if pts_c.size == 0:
                    continue

                eps = self.dbscan_eps_map.get(sem_id, 1.0)
                clusters = cluster_points_dbscan(pts_c, eps=eps, min_samples=5)

                for cl in clusters:
                    # Transform cluster points from LiDAR to rect camera coordinate
                    cl_rect = lidar_to_rect(cl, R0, Tr)
                    x_min, y_min, z_min = cl_rect.min(axis=0)
                    x_max, y_max, z_max = cl_rect.max(axis=0)

                    # Dimensions: l (length=Δz), h (height=Δy), w (width=Δx)
                    w = max(0.1, z_max - z_min)
                    h = max(0.1, y_max - y_min)
                    l = max(0.1, x_max - x_min)

                    # Center point: x, z take midpoint; y take bottom (y_max, KITTI convention)
                    x_c = (x_min + x_max) / 2.0
                    y_c = y_max  # Note: KITTI's y is height direction, bottom is y_max
                    z_c = (z_min + z_max) / 2.0

                    boxes.append((name, h, w, l, float(x_c), float(y_c), float(z_c), 0.0))

            # Write label_2 file
            write_label2(output_path, boxes)
            return True
            
        except Exception as e:
            print(f"[ERROR] Frame {frame_id}: {e}")
            return False

