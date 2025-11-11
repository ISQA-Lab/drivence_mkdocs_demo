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
    """Read velodyne bin file and return points (N, 3)."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Invalid velodyne bin file: {path}")
    pts = data.reshape((-1, 4))[:, :3]
    return pts


def read_semantic_label(path: str) -> np.ndarray:
    """Read semantic label file (.label or .npy)."""
    if path.endswith(".label"):
        labels = np.fromfile(path, dtype=np.uint32)
        semantic = labels & 0xFFFF
        return semantic.astype(np.int32)
    elif path.endswith(".npy"):
        return np.load(path).astype(np.int32).reshape(-1)
    else:
        raise ValueError(f"Unsupported semantic label file: {path}")


def cluster_points_dbscan(points: np.ndarray, eps: float, min_samples: int = 5) -> list:
    """Cluster points in XY plane using DBSCAN, return list of point clusters."""
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
    """Write KITTI-format label_2 file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name, h, w, l, x, y, z, ry in boxes:
            f.write(f"{name} -1 -1 0 0 0 0 0 {h:.3f} {w:.3f} {l:.3f} {x:.3f} {y:.3f} {z:.3f} {ry:.3f}\n")


def read_kitti_calib(calib_txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read KITTI calibration file and return R0 and Tr."""
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
    """Transform points from LiDAR to rect camera coordinate."""
    ones = np.ones((points_lidar.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([points_lidar.astype(np.float32), ones], axis=1)  # (N,4)
    pts_cam = (Tr @ pts_h.T).T  # (N,3)
    pts_rect = (R0 @ pts_cam.T).T  # (N,3)
    return pts_rect


def rect_to_lidar(points_rect: np.ndarray, R0: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """Transform points from rect camera coordinate to LiDAR coordinate.
    
    Inverse of lidar_to_rect.
    
    Args:
        points_rect: Points in rect coordinate (N, 3)
        R0: Rectification matrix (3, 3)
        Tr: Transformation matrix from LiDAR to camera (3, 4)
    
    Returns:
        Points in LiDAR coordinate (N, 3)
    """
    # Step 1: rect -> cam: cam = R0^T @ rect (R0 is orthogonal)
    if len(points_rect.shape) == 1:
        points_rect = points_rect.reshape(1, -1)
    points_cam = (R0.T @ points_rect.T).T  # (N, 3)
    
    # Step 2: cam -> velo: need inverse of Tr
    # Tr is 3x4: [R | t], where cam = R @ velo + t
    # So: velo = R^-1 @ (cam - t) = R^T @ (cam - t) (R is rotation, so R^-1 = R^T)
    R = Tr[:, :3]  # 3x3 rotation matrix
    t = Tr[:, 3]    # 3x1 translation vector
    
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
        Generate label_2 file from semantic label for a single frame.
        
        Args:
            frame_id: Frame ID (for logging)
            velodyne_path: Path to velodyne bin file
            semantic_label_path: Path to semantic label file (.label or .npy)
            calib_path: Path to calibration file
            output_path: Output path for label_2 file
            
        Returns:
            True if successful, False otherwise
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

