import os
from typing import Tuple, List

import numpy as np
import numpy

from drivence.utils.path_utils import get_project_root_dir

def load_road_split_labels(label_path: str) -> list:
    labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    return list(labels)


class RoadSplit(object):
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def get_pc_road_in_img(self, pts_img: numpy.ndarray, pts_rect_depth: numpy.ndarray,
                           points: numpy.ndarray) -> numpy.ndarray:

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
        
        if os.path.exists(pre_extracted_bin_path) and os.path.exists(pre_extracted_label_path):
            print(f"Loading pre-extracted {insert_location} points from: {pre_extracted_bin_path}")
            pc_road = np.fromfile(pre_extracted_bin_path, dtype=np.float32).reshape((-1, 3))
            if len(pc_road) <= 10:
                return None, None
            
            non_road_bin_path = os.path.join(pre_extracted_dir, f"{bg_index:06d}_non_road.bin")
            if os.path.exists(non_road_bin_path):
                _pc_non_road = np.fromfile(non_road_bin_path, dtype=np.float32).reshape((-1, 3))
                print(f"Loading cached non-road points from: {non_road_bin_path}")
                return pc_road, _pc_non_road
            else:
                labels = load_road_split_labels(pre_extracted_label_path)
                pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                road_index_arr, inx_no_road_arr = self.split_pcd_road_label(labels, insert_location)
                _pc_non_road = pc_bg[inx_no_road_arr]
                _pc_non_road.astype(np.float32).tofile(non_road_bin_path)
                print(f"Cached non-road points to: {non_road_bin_path}")
                return pc_road, _pc_non_road
        
        if os.path.exists(save_road_interpolation_path) and os.path.exists(save_road_label_path):
            print(f"Loading cached {insert_location} points from: {save_road_interpolation_path}")
            pc_road = np.fromfile(save_road_interpolation_path, dtype=np.float32).reshape((-1, 3))
            if len(pc_road) <= 10:
                return None, None
            
            non_road_bin_path = os.path.join(save_road_label_dir, f"{bg_index:06d}_non_road.bin")
            if os.path.exists(non_road_bin_path):
                _pc_non_road = np.fromfile(non_road_bin_path, dtype=np.float32).reshape((-1, 3))
                print(f"Loading cached non-road points from: {non_road_bin_path}")
                return pc_road, _pc_non_road
            labels = load_road_split_labels(save_road_label_path)
            pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
            road_index_arr, inx_no_road_arr = self.split_pcd_road_label(labels, insert_location)
            _pc_non_road = pc_bg[inx_no_road_arr]
            _pc_non_road.astype(np.float32).tofile(non_road_bin_path)
            print(f"Cached non-road points to: {non_road_bin_path}")
            return pc_road, _pc_non_road
        
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
        
        if semantic_label_dir is None or not os.path.exists(semantic_label_dir):
            print(f"Warning: GT semantic label directory not found. Cannot auto-generate {insert_location} points.")
            print(f"Please ensure pre-extracted files exist in: {pre_extracted_dir}")
            return None, None
        
        print(f"Auto-generating {insert_location} points from GT labels in: {semantic_label_dir}")
        pc_road, labels = self._generate_from_gt_labels(bg_index, bg_pc_path, semantic_label_dir, insert_location)
        
        if pc_road is None or len(pc_road) <= 10:
            print(f"Warning: Generated {insert_location} points are too few or None")
            return None, None
        
        os.makedirs(pre_extracted_dir, exist_ok=True)
        pre_extracted_label_path = os.path.join(pre_extracted_dir, f"{bg_index:06d}.label")
        pre_extracted_bin_path = os.path.join(pre_extracted_dir, f"{bg_index:06d}.bin")
        
        labels.astype(np.uint32).tofile(pre_extracted_label_path)
        pc_road.astype(np.float32).tofile(pre_extracted_bin_path)
        print(f"Saved generated {insert_location} points to: {pre_extracted_bin_path}")
        
        labels.astype(np.uint32).tofile(save_road_label_path)
        pc_road.astype(np.float32).tofile(save_road_interpolation_path)
        
        pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        road_index_arr, inx_no_road_arr = self.split_pcd_road_label(labels.reshape(-1, 1).tolist(), insert_location)
        _pc_non_road = pc_bg[inx_no_road_arr]

        non_road_bin_path_pre = os.path.join(pre_extracted_dir, f"{bg_index:06d}_non_road.bin")
        non_road_bin_path_cache = os.path.join(save_road_label_dir, f"{bg_index:06d}_non_road.bin")
        _pc_non_road.astype(np.float32).tofile(non_road_bin_path_pre)
        _pc_non_road.astype(np.float32).tofile(non_road_bin_path_cache)

        return pc_road, _pc_non_road

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
