import os
from typing import Tuple, List, Optional

import numpy as np
import numpy

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - optional dependency
    cKDTree = None

from drivence.utils.path_utils import get_project_root_dir

def load_road_split_labels(label_path: str) -> list:
    labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    return list(labels)


class RoadSplit(object):
    """
    路面点云分割基类，定义路面点云处理的核心接口和基础方法。

    核心功能：
    1. 提供图像视野（FOV）内路面点云的筛选方法；
    2. 定义路面点云与非路面点云的分割接口（需子类实现具体分割逻辑）；
    3. 适配不同场景的点云输入格式，支持语义标签辅助分割（可选）。
    """
    def __init__(self, img_height, img_width):
        """
        初始化路面分割器，配置图像尺寸（用于FOV筛选）。

        Args:
            img_height (int): 图像高度（像素），对应点云投影后的Y轴范围；
            img_width (int): 图像宽度（像素），对应点云投影后的X轴范围。
        """
        self.img_height = img_height
        self.img_width = img_width

    def get_pc_road_in_img(self, pts_img: numpy.ndarray, pts_rect_depth: numpy.ndarray,
                           points: numpy.ndarray) -> numpy.ndarray:
        """
        筛选出位于图像视野（FOV）内的点云（基于点云投影后的图像坐标和深度）。

        核心逻辑：
        1. 校验点云投影后的图像坐标是否在图像尺寸范围内（X∈[0, img_width)，Y∈[0, img_height)）；
        2. 校验点云深度是否有效（≥0）；
        3. 基于上述条件筛选出有效点云，返回视野内的点云数据。

        Args:
            pts_img (numpy.ndarray): 点云投影到图像平面的坐标数组，形状为 (N, 2)，
                每行对应 [u, v]（图像X、Y像素坐标）；
            pts_rect_depth (numpy.ndarray): 点云的深度数组（矩形坐标系下），形状为 (N,)，
                每个元素为对应点的深度值（需≥0才有效）；
            points (numpy.ndarray): 原始3D点云数组，形状为 (N, 3)，每行对应 [x, y, z] 坐标。

        Returns:
            numpy.ndarray: 图像视野内的有效点云数组，形状为 (M, 3)，M≤N。
        """
        assert points.shape[1] == 3
        img_shape = (self.img_height, self.img_width)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        pts_fov = points[pts_valid_flag]
        return pts_fov

    def split_pcd_road(self, bg_index: int, bg_pc_path: str, save_road_label_dir: str, log_dir: str,
                       insert_location="road", semantic_label_dir=None) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        分割背景点云为路面点云和非路面点云（基类抽象接口，需子类实现具体分割逻辑）。

        子类需根据场景需求实现分割算法（如基于语义标签、高程阈值、地面拟合等），
        返回分割后的路面点云和非路面点云，支持保存分割结果和日志。

        Args:
            bg_index (int): 背景索引（如数据集帧号、场景序列号），用于标识当前处理的背景；
            bg_pc_path (str): 背景点云文件路径（绝对/相对路径），用于读取原始背景点云；
            save_road_label_dir (str): 路面分割结果保存目录（如保存路面点云、语义标签文件）；
            log_dir (str): 日志保存目录（用于记录分割过程中的关键信息、异常日志）；
            insert_location (str, 可选): 插入位置标识（默认 "road"，用于区分分割场景，如路面、人行道等）；
            semantic_label_dir (Optional[str], 可选): 语义标签文件目录（若基于语义标签分割，需读取该目录下的标签文件），
                默认为 None（不使用语义标签辅助分割）。

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: 分割结果，格式为 (road_pc, non_road_pc)：
                - road_pc: 路面点云数组，形状为 (M, 3+)；
                - non_road_pc: 非路面点云数组，形状为 (K, 3+)。
        """
        raise NotImplementedError("RoadSplit.split_pcd_road must be implemented by subclasses")


class SemanticKITTIRoadSplit(RoadSplit):
    """
    Road split for SemanticKITTI dataset.
    Supports loading pre-extracted road/sidewalk/terrain points or generating from GT labels.
    """
    def __init__(self, img_height, img_width):
        super().__init__(img_height, img_width)

    def split_pcd_road(self, bg_index: int, bg_pc_path: str, save_road_label_dir: str, log_dir: str, 
                       insert_location="road", semantic_label_dir=None) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Split point cloud into road and non-road points.
        
        Args:
            bg_index: Background frame index
            bg_pc_path: Path to background point cloud file
            save_road_label_dir: Directory to save road split labels
            log_dir: Directory for log files
            insert_location: Type of surface to extract ("road", "sidewalk", "terrain")
            semantic_label_dir: Directory containing GT semantic labels (for auto-generation)
        
        Returns:
            Tuple of (road_points, non_road_points)
        """
        location_to_dir = {
            "road": "road_split_label",
            "sidewalk": "sidewalk_split_label",
            "terrain": "terrain_split_label"
        }
        pre_extracted_dirname = location_to_dir.get(insert_location, "road_split_label")
        pre_extracted_dir = os.path.join(os.path.dirname(save_road_label_dir), pre_extracted_dirname)
        
        pre_extracted_bin_path = os.path.join(pre_extracted_dir, f"{bg_index:06d}.bin")
        pre_extracted_label_path = os.path.join(pre_extracted_dir, f"{bg_index:06d}.label")
        
        save_road_label_path = f"{save_road_label_dir}/{bg_index:06d}.label"
        save_road_interpolation_path = f"{save_road_label_dir}/{bg_index:06d}.bin"
        
        os.makedirs(save_road_label_dir, exist_ok=True)
        os.makedirs(pre_extracted_dir, exist_ok=True)
        
        reference_pc: Optional[np.ndarray] = None
        reference_labels: Optional[np.ndarray] = None

        if os.path.exists(pre_extracted_bin_path) and os.path.exists(pre_extracted_label_path):
            print(f"Loading pre-extracted {insert_location} points from: {pre_extracted_bin_path}")
            reference_pc = np.fromfile(pre_extracted_bin_path, dtype=np.float32).reshape((-1, 3))
            reference_labels = np.fromfile(pre_extracted_label_path, dtype=np.uint32).reshape((-1, 1))
        elif os.path.exists(save_road_interpolation_path) and os.path.exists(save_road_label_path):
            print(f"Loading cached {insert_location} points from: {save_road_interpolation_path}")
            reference_pc = np.fromfile(save_road_interpolation_path, dtype=np.float32).reshape((-1, 3))
            reference_labels = np.fromfile(save_road_label_path, dtype=np.uint32).reshape((-1, 1))

        if semantic_label_dir is None:
            dataset_dir = os.path.dirname(os.path.dirname(bg_pc_path))
            possible_label_dirs = [
                os.path.join(dataset_dir, "labels"),
                os.path.join(dataset_dir, "semantic_label"),
                os.path.join(os.path.dirname(bg_pc_path), "..", "labels"),
                os.path.join(os.path.dirname(bg_pc_path), "..", "semantic_label"),
            ]
            semantic_label_dir = None
            for label_dir in possible_label_dirs:
                test_label_path = os.path.join(label_dir, f"{bg_index:06d}.label")
                if os.path.exists(test_label_path):
                    semantic_label_dir = label_dir
                    break
        
        if reference_pc is None or len(reference_pc) <= 10:
            if semantic_label_dir is None or not os.path.exists(semantic_label_dir):
                print(f"Warning: GT semantic label directory not found. Cannot auto-generate {insert_location} points.")
                print(f"Please ensure pre-extracted files exist in: {pre_extracted_dir}")
                return None, None

            print(f"Auto-generating {insert_location} points from GT labels in: {semantic_label_dir}")
            generated_pc, labels = self._generate_from_gt_labels(
                bg_index, bg_pc_path, semantic_label_dir, insert_location
            )

            if generated_pc is None or len(generated_pc) <= 10:
                print(f"Warning: Generated {insert_location} points are too few or None")
                return None, None

            reference_pc = generated_pc
            reference_labels = labels.reshape((-1, 1))

            os.makedirs(pre_extracted_dir, exist_ok=True)
            reference_labels.astype(np.uint32).tofile(pre_extracted_label_path)
            reference_pc.astype(np.float32).tofile(pre_extracted_bin_path)
            reference_labels.astype(np.uint32).tofile(save_road_label_path)
            reference_pc.astype(np.float32).tofile(save_road_interpolation_path)
            print(f"Saved generated {insert_location} points to: {pre_extracted_bin_path}")

        if reference_pc is None or len(reference_pc) <= 10:
            return None, None

        pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        road_pc, non_road_pc = self._split_by_distance(pc_bg, reference_pc, insert_location=insert_location)
        if road_pc is None or len(road_pc) == 0:
            print(f"Warning: Distance-based split failed for {insert_location}")
            return None, None

        road_pc.astype(np.float32).tofile(save_road_interpolation_path)
        non_road_bin_path_cache = os.path.join(save_road_label_dir, f"{bg_index:06d}_non_road.bin")
        non_road_pc.astype(np.float32).tofile(non_road_bin_path_cache)

        if os.path.isdir(pre_extracted_dir):
            non_road_bin_path_pre = os.path.join(pre_extracted_dir, f"{bg_index:06d}_non_road.bin")
            non_road_pc.astype(np.float32).tofile(non_road_bin_path_pre)

        return road_pc, non_road_pc

    def _generate_from_gt_labels(self, bg_index: int, bg_pc_path: str, semantic_label_dir: str, 
                                  insert_location: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Generate road/sidewalk/terrain points from GT semantic labels.
        
        Args:
            bg_index: Background frame index
            bg_pc_path: Path to background point cloud file
            semantic_label_dir: Directory containing GT semantic labels
            insert_location: Type of surface to extract ("road", "sidewalk", "terrain")
        
        Returns:
            Tuple of (processed_points, original_labels)
        """
        location_to_label = {
            "road": 40,
            "sidewalk": 48,
            "terrain": 72
        }
        target_label = location_to_label.get(insert_location, 40)
        
        points = np.fromfile(bg_pc_path, dtype=np.float32)
        points = points.reshape(-1, 4)
        xyz = points[:, :3]
        
        label_path = os.path.join(semantic_label_dir, f"{bg_index:06d}.label")
        if not os.path.exists(label_path):
            print(f"Error: GT label file not found: {label_path}")
            return None, None
        
        labels = np.fromfile(label_path, dtype=np.uint32)
        
        if len(xyz) != len(labels):
            print(f"Error: Point cloud and label count mismatch: {len(xyz)} vs {len(labels)}")
            return None, None
        
        target_mask = (labels == target_label)
        target_points = xyz[target_mask]
        
        if len(target_points) <= 10:
            print(f"Warning: Too few {insert_location} points found: {len(target_points)}")
            return None, None
        
        print(f"Generated {insert_location} points: {len(target_points)}")
        
        return target_points.astype(np.float32), labels

    def _split_by_distance(self, pc_bg: np.ndarray, reference_pc: np.ndarray,
                           insert_location: str = "road",
                           distance_threshold: float = 0.3) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if reference_pc is None or len(reference_pc) == 0:
            return None, None

        if cKDTree is None:
            raise RuntimeError(
                "scipy is required for distance-based road splitting. Please install scipy."
            )

        if pc_bg.ndim != 2 or pc_bg.shape[1] != 3:
            pc_bg = pc_bg.reshape(-1, 3)

        # Allow different thresholds per surface if needed later.
        surface_thresholds = {
            "road": distance_threshold,
            "sidewalk": distance_threshold,
            "terrain": distance_threshold,
        }
        threshold = surface_thresholds.get(insert_location, distance_threshold)

        tree = cKDTree(reference_pc)
        distances, _ = tree.query(pc_bg, k=1)

        road_mask = distances < threshold
        road_points = pc_bg[road_mask]
        non_road_points = pc_bg[~road_mask]

        print(
            f"Distance-based split ({insert_location}): "
            f"road_points={len(road_points)}, non_road_points={len(non_road_points)}"
        )

        return road_points.astype(np.float32), non_road_points.astype(np.float32)
    

    def split_pcd_road_label(self, labels: list, insert_location="road") -> Tuple[List, List]:

        inx_road_arr = []
        inx_other_road_arr = []
        inx_other_ground_arr = []
        inx_no_road_arr = []

        for i in range(len(labels)):
            inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = self._split_pc_road_label_detail(
                labels, i,
                inx_road_arr,
                inx_other_road_arr,
                inx_other_ground_arr,
                inx_no_road_arr,
                insert_location
            )
        return inx_road_arr, [*inx_other_road_arr, *inx_other_ground_arr, *inx_no_road_arr]

    def _split_pc_road_label_detail(self, label, index, inx_road_arr, inx_other_road_arr, inx_other_ground_arr,
                                    inx_no_road_arr, insert_location="road"):

        lb = label[index][0]
        if insert_location == "road":
            if lb == 40:
                inx_road_arr.append(index)
            elif lb == 44:
                inx_other_road_arr.append(index)
            elif lb == 48:
                inx_other_road_arr.append(index)
            elif lb in (49, 70, 71, 72):
                inx_other_ground_arr.append(index)
            else:
                inx_no_road_arr.append(index)
            return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr
        elif insert_location == "sidewalk":
            if lb == 48:
                inx_road_arr.append(index)
            elif lb == 44:
                inx_other_road_arr.append(index)
            elif lb == 40:
                inx_other_road_arr.append(index)
            elif lb in (49, 70, 71, 72):
                inx_other_ground_arr.append(index)
            else:
                inx_no_road_arr.append(index)
            return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr
        elif insert_location == "terrain":
            if lb == 72:
                inx_road_arr.append(index)
            elif lb == 44:
                inx_other_road_arr.append(index)
            elif lb == 40:
                inx_other_road_arr.append(index)
            elif lb == 48:
                inx_other_road_arr.append(index)
            elif lb in (49, 70, 71, 72):
                inx_other_ground_arr.append(index)
            else:
                inx_no_road_arr.append(index)
            return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr

        return None


if __name__ == '__main__':
    ...
