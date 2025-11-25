
import time
import numpy as np
import uuid
import os
from typing import Optional


def extract_initial_objs_from_bg(calib_info, label, label_2_path: Optional[str] = None):
    """
    从背景场景中提取物体的3D包围盒（LiDAR坐标系下），用于后续碰撞检测。

    核心逻辑：
    1. 优先使用 SemanticKITTI 格式的 label_2 文件（若路径有效），通过 `extract_objs_from_label2` 提取包围盒；
    2. 若未提供 label_2 文件，则使用原始 KITTI 格式的标签对象，遍历每个物体并提取其 LiDAR 坐标系下的3D角点；
    3. 处理空输入场景（无标签/无有效文件），返回形状为 (0, 8, 3) 的空数组，确保输出格式一致性。

    Args:
        calib_info (Optional[dict]): 标定信息字典（包含相机-激光雷达外参等，用于坐标转换），
            SemanticKITTI 场景下可为 None；
        label (Optional[List[KITTIObject]]): 原始 KITTI 格式的标签对象列表，每个元素为一个物体的标签信息，
            需支持 `get_box3d_corners_in_lidar_coord` 方法（提取LiDAR坐标系下的3D角点），
            SemanticKITTI 场景下可为 None；
        label_2_path (Optional[str], 可选): SemanticKITTI 格式的 label_2 文件路径（文本文件），
            用于 SemanticKITTI 场景下提取物体包围盒，默认 None。

    Returns:
        np.ndarray: 背景物体的3D包围盒角点数组，形状为 (N, 8, 3)：
            - N：背景物体数量（无物体时 N=0）；
            - 8：每个3D包围盒的8个角点；
            - 3：每个角点的 X/Y/Z 坐标（LiDAR坐标系下）；
            若无背景物体，返回形状为 (0, 8, 3) 的空数组（避免后续处理维度错误）。
    """
    # If label_2_path is provided, use it for SemanticKITTI
    if label_2_path is not None and os.path.exists(label_2_path):
        return extract_objs_from_label2(label_2_path, calib_info)
    
    # Otherwise, use original KITTI label format
    if label is None:
        return np.array([]).reshape(0, 8, 3)
    bg_objs_bounding_box = []
    for obj in label:
        bg_obj_bounding_box = obj.get_box3d_corners_in_lidar_coord(calib_info)
        bg_objs_bounding_box.append(bg_obj_bounding_box)
    return np.asarray(bg_objs_bounding_box) if bg_objs_bounding_box else np.array([]).reshape(0, 8, 3)


def extract_objs_from_label2(label_2_path: str, calib_info) -> np.ndarray:
    """
    从 KITTI 格式的 label_2 文本文件中提取物体的3D包围盒，并转换为 LiDAR 坐标系下的8个角点。

    核心逻辑：
    1. 校验 label_2 文件有效性，读取文件中每行的物体标注信息；
    2. 若未提供标定信息，尝试从数据集目录中自动推断并读取 calib.txt（KITTI 标定文件）；
    3. 解析 label_2 每行的物体参数（尺寸、矩形坐标系中心、朝向角）；
    4. 将矩形坐标系（rect）下的物体中心转换为 LiDAR 坐标系；
    5. 构建 LiDAR 坐标系下的7维包围盒（中心+尺寸+朝向），并转换为8个3D角点。

    Args:
        label_2_path (str): KITTI 格式 label_2 文件的路径（绝对路径或相对路径），
            文件每行存储一个物体的标注信息，格式遵循 KITTI 标准；
        calib_info (Optional[dict], 可选): 标定信息字典（包含 R0、Tr 等外参），
            若为 None，将尝试从 label_2 文件所在目录的上级目录中自动查找 calib.txt，默认 None。

    Returns:
        np.ndarray: LiDAR 坐标系下的物体3D包围盒角点数组，形状为 (N, 8, 3)：
            - N：有效提取的物体数量（无有效物体时 N=0）；
            - 8：每个3D包围盒的8个角点；
            - 3：每个角点的 X/Y/Z 坐标（LiDAR 坐标系下）；
            若无有效物体或文件读取失败，返回形状为 (0, 8, 3) 的空数组，确保输出格式一致性。
    """
    from drivence.utils.box_utils import lidar_boxesn7_to_corners3d
    from drivence.module.label_2_generator.label_2_generator import read_kitti_calib, rect_to_lidar
    
    if not os.path.exists(label_2_path):
        return np.array([]).reshape(0, 8, 3)
    
    # Try to get calib info from dataset if not provided
    calib_path = None
    if calib_info is None:
        # Try to infer calib path from label_2_path
        dataset_dir = os.path.dirname(os.path.dirname(label_2_path))
        calib_path = os.path.join(dataset_dir, "calib.txt")
        if not os.path.exists(calib_path):
            calib_path = os.path.join(dataset_dir, "calib", "calib.txt")
        if not os.path.exists(calib_path):
            calib_path = None
    
    # Read calibration if available
    R0 = None
    Tr = None
    if calib_path and os.path.exists(calib_path):
        try:
            R0, Tr = read_kitti_calib(calib_path)
        except Exception as e:
            print(f"[WARN] Failed to read calib file {calib_path}: {e}")
    
    boxes = []
    with open(label_2_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            
            # KITTI format: type truncated occluded alpha bbox_2d h w l x y z ry
            # parts[0] = type, parts[8] = h, parts[9] = w, parts[10] = l
            # parts[11] = x, parts[12] = y, parts[13] = z, parts[14] = ry
            try:
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                x_rect = float(parts[11])
                y_rect = float(parts[12])
                z_rect = float(parts[13])
                ry = float(parts[14])
            except (ValueError, IndexError):
                continue
            
            # Convert from rect to LiDAR coordinate
            # Note: label_2 format uses rect coordinate (x, y, z, h, w, l, ry)
            # Use the same transformation as gen_label2_from_semantic.py (inverse)
            if R0 is not None and Tr is not None:
                # Use rect_to_lidar function (inverse of lidar_to_rect from gen_label2_from_semantic.py)
                center_rect = np.array([x_rect, y_rect, z_rect], dtype=np.float32)
                center_lidar = rect_to_lidar(center_rect, R0, Tr)
                x_lidar, y_lidar, z_lidar = center_lidar
            else:
                # Use rect coordinates directly (approximation)
                # Note: This is not accurate but may work for collision detection
                x_lidar, y_lidar, z_lidar = x_rect, y_rect, z_rect
            
            # Box format: [x, y, z, dx, dy, dz, heading] in LiDAR coordinate
            # Note: dimensions (h, w, l) in rect may need adjustment, but for collision detection
            # we'll use them as-is (approximation)
            box_lidar = np.array([x_lidar, y_lidar, z_lidar, l, w, h, ry])
            boxes.append(box_lidar)
    
    if len(boxes) == 0:
        return np.array([]).reshape(0, 8, 3)
    
    # Convert boxes to corners
    boxes_n7 = np.array(boxes)  # (N, 7)
    corners = lidar_boxesn7_to_corners3d(boxes_n7)  # (N, 8, 3)
    return corners
# Singleton helper
class UUIDManager:
    _instance = None  # Holds the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UUIDManager, cls).__new__(cls)
            cls._instance._uuid = str(uuid.uuid4())  # Initialize UUID
        return cls._instance

    def get_uuid(self, reset=False):
        """Return the current UUID, optionally generating a new one."""
        if reset:
            self._uuid = str(uuid.uuid4())
        return self._uuid

    def reset_uuid(self, new_uuid=None):
        """Set a new UUID explicitly or generate one if not provided."""
        self._uuid = str(new_uuid) if new_uuid else str(uuid.uuid4())

    @classmethod
    def get_instance(cls):
        """Retrieve the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# python -m drivence.utils.common_utils
if __name__ == "__main__":
    uuid_manager = UUIDManager()
    print(uuid_manager.get_uuid())
    time.sleep(1)
    print(uuid_manager.get_uuid())
    uuid_manager.reset_uuid()
    print(uuid_manager.get_uuid())
    print(uuid_manager.get_uuid())
    print(uuid_manager.get_uuid(reset=True))
    print(uuid_manager.get_uuid())
    print(uuid_manager.get_uuid())
    uuid_manager = UUIDManager()
    print(uuid_manager.get_uuid())
    uuid_manager.get_instance()
    print(uuid_manager.get_uuid())
