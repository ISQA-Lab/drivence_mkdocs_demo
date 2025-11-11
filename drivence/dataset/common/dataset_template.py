import os

import numpy as np
from drivence.utils.pc_utils import PointCloud

class DataFrameTemplate(object):
    # @Test status: Completed
    def __init__(self, frame_id, dataset_config: dict):
        if "split" in dataset_config.keys():
            self.dataset_dir = os.path.join(dataset_config["root_dir"], dataset_config["split"])
        else:
            self.dataset_dir = dataset_config["root_dir"]
        self.image_dir = None  # init in 子类
        self.point_cloud_dir = None
        self.calib_dir = None
        self.label_dir = None
        self.depth_dir = None
        self.road_dir = None
        self.dataset_config = dataset_config
        self.index = frame_id
        self.index_str = f"{self.index:06d}"

        self.road_split_label_dir = None
        self.road_split_log_dir = None
        self.lidar_scene_info_dir = None
        self.camera_scene_info_dir = None
        self.init_dir()

    def init_dir(self):
        raise NotImplementedError("DataFrameTemplate.init_dir must be implemented by subclasses")

    def get_road_path(self) -> str:
        road_filepath = os.path.join(self.road_dir, f"{self.index_str}.bin")
        return road_filepath

    def get_image_path(self) -> str:
        image_filepath = os.path.join(self.image_dir, f"{self.index_str}.png")
        return image_filepath

    def get_depth_path(self) -> str:
        depth_filepath = os.path.join(self.depth_dir, f"{self.index_str}.png")
        return depth_filepath

    def get_point_cloud_path(self) -> str:
        point_cloud_filepath = os.path.join(self.point_cloud_dir, f"{self.index_str}.bin")
        return point_cloud_filepath

    def get_calib_path(self) -> str:
        calib_filepath = os.path.join(self.calib_dir, f"{self.index_str}.txt")
        return calib_filepath

    def get_label_path(self) -> str:
        label_filepath = os.path.join(self.label_dir, f"{self.index_str}.txt")
        return label_filepath

    def get_road_label_path(self) -> str:
        road_label_filepath = os.path.join(self.label_dir, f"{self.index_str}.txt")
        return road_label_filepath

    def get_lidar_scene_info_path(self) -> str:
        lidar_scene_info_filepath = os.path.join(self.lidar_scene_info_dir, f"{self.index_str}.json")
        return lidar_scene_info_filepath

    def get_camera_scene_info_path(self) -> str:
        camera_scene_info_filepath = os.path.join(self.camera_scene_info_dir, f"{self.index_str}.json")
        return camera_scene_info_filepath

    def get_depth(self) -> np.ndarray:
        raise NotImplementedError("DataFrameTemplate.get_depth must be implemented by subclasses")

    def get_image(self) -> np.ndarray:
        raise NotImplementedError("DataFrameTemplate.get_image must be implemented by subclasses")

    def get_point_cloud(self) -> PointCloud:
        raise NotImplementedError("DataFrameTemplate.get_point_cloud must be implemented by subclasses")

    def get_calibration(self):
        raise NotImplementedError("DataFrameTemplate.get_calibration must be implemented by subclasses")

    def get_label(self):
        raise NotImplementedError("DataFrameTemplate.get_label must be implemented by subclasses")

    def save(self):
        raise NotImplementedError("DataFrameTemplate.save must be implemented by subclasses")
