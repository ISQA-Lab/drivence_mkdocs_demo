
import time
import numpy as np
import uuid
import os
from typing import Optional


def extract_initial_objs_from_bg(calib_info, label, label_2_path: Optional[str] = None):
    """Extract background objects bounding boxes for collision detection.
    
    Args:
        calib_info: Calibration info (can be None for SemanticKITTI)
        label: KITTI label object (can be None for SemanticKITTI)
        label_2_path: Path to label_2 file (used for SemanticKITTI when label is None)
    
    Returns:
        Background objects bounding boxes (N, 8, 3) in LiDAR coordinate
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
    """Extract bounding boxes from KITTI-format label_2 file.
    
    Args:
        label_2_path: Path to label_2 file
        calib_info: Calibration info (can be None, will try to read from dataset)
    
    Returns:
        Background objects bounding boxes (N, 8, 3) in LiDAR coordinate
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
