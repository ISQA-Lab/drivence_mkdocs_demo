import os
import numpy as np

from drivence.dataset.common.dataset_template import DataFrameTemplate
from drivence.dataset.common.point_cloud import PointCloud
from drivence.utils import pc_utils
from drivence.utils.path_utils import get_project_root_dir


class SemanticKITTIDetectionDataFrame(DataFrameTemplate):
    """
    SemanticKITTI DataFrame
    Directory layout example (set split to "sequences/08"):
      root_dir/
        sequences/08/
          velodyne/000000.bin
          labels/  000000.label
    """

    def __init__(self, frame_id: int, dataset_config: dict):
        super().__init__(frame_id, dataset_config)

    def init_dir(self):
        # For SemanticKITTI, recommend setting split to e.g. "sequences/08"
        if "point_cloud_dirname" in self.dataset_config:
            self.point_cloud_dir = os.path.join(self.dataset_dir, self.dataset_config["point_cloud_dirname"])
        else:
            self.point_cloud_dir = os.path.join(self.dataset_dir, "velodyne")

        # Semantic labels (.label) - respect config directory name if provided
        sem3d_dirname = self.dataset_config.get("semantic_label_3d", "labels")
        self.semantic_label_dir = os.path.join(self.dataset_dir, sem3d_dirname)

        # Optional extras commonly used in pipeline
        if "road_split_label" in self.dataset_config:
            self.road_split_label_dir = os.path.join(self.dataset_dir, self.dataset_config["road_split_label"])
        if "road_split_log" in self.dataset_config:
            self.road_split_log_dir = os.path.join(self.dataset_dir, self.dataset_config["road_split_log"])
        if "lidar_scene_info" in self.dataset_config:
            self.lidar_scene_info_dir = os.path.join(self.dataset_dir, self.dataset_config["lidar_scene_info"])
        if "label_2" in self.dataset_config:
            self.label_2_dir = os.path.join(self.dataset_dir, self.dataset_config["label_2"])
        else:
            self.label_2_dir = os.path.join(self.dataset_dir, "label_2")
        
        # Calibration file for label_2 generation
        calib_path = os.path.join(self.dataset_dir, "calib.txt")
        if not os.path.exists(calib_path):
            calib_path = os.path.join(self.dataset_dir, "calib", "calib.txt")
        self.calib_path = calib_path if os.path.exists(calib_path) else None

        # No image/calib/depth for pc-only
        self.image_dir = None
        self.calib_dir = None
        self.depth_dir = None
        self.label_dir = None

    # ------- pc-only APIs -------
    def get_point_cloud(self) -> PointCloud:
        point_cloud_filepath = os.path.join(self.point_cloud_dir, f"{self.index_str}.bin")
        point_cloud_npy = pc_utils.load_pc(point_cloud_filepath, demension=4)
        pc = PointCloud()
        pc.set_point_cloud(point_cloud_npy[:, :3])
        pc.set_reflection(point_cloud_npy[:, 3])
        return pc

    def get_semantic_label_3d(self) -> np.ndarray:
        """Load SemanticKITTI .label (uint32). Caller can remap as needed."""
        label_path = os.path.join(self.semantic_label_dir, f"{self.index_str}.label")
        if not os.path.exists(label_path):
            print(f"Warning: semantic label file not found: {label_path}")
            return None
        labels = np.fromfile(label_path, dtype=np.uint32)
        return labels

    # ------- Unused/pc-only stubs -------
    def get_image(self):
        return None

    def get_depth(self):
        return None

    def get_calibration(self):
        return None

    def get_label(self):
        # No 2D/3D detection labels in SemanticKITTI baseline
        return None
    
    def get_label_2_path(self) -> str:
        """Get path to label_2 file (KITTI-format for collision detection)."""
        return os.path.join(self.label_2_dir, f"{self.index_str}.txt")
    
    def get_label_2(self) -> str:
        """
        Get label_2 file path, generating it if it doesn't exist.
        
        Returns:
            Path to label_2 file
        """
        label_2_path = self.get_label_2_path()
        
        # If file exists, return it
        if os.path.exists(label_2_path):
            return label_2_path
        
        # Otherwise, generate it from semantic labels
        from drivence.module.label_2_generator.label_2_generator import SemanticKITTILabel2Generator
        
        generator = SemanticKITTILabel2Generator()
        
        # Get paths
        velodyne_path = os.path.join(self.point_cloud_dir, f"{self.index_str}.bin")
        semantic_label_path = os.path.join(self.semantic_label_dir, f"{self.index_str}.label")
        
        # Check if semantic label exists
        if not os.path.exists(semantic_label_path):
            # Try .npy format
            semantic_label_path = os.path.join(self.semantic_label_dir, f"{self.index_str}.npy")
            if not os.path.exists(semantic_label_path):
                print(f"[WARN] No semantic labels for frame {self.index}, cannot generate label_2")
                return label_2_path
        
        # Check if calib file exists
        if self.calib_path is None or not os.path.exists(self.calib_path):
            print(f"[WARN] No calibration file found, cannot generate label_2")
            return label_2_path
        
        # Generate label_2 file
        os.makedirs(self.label_2_dir, exist_ok=True)
        success = generator.generate_label2_from_semantic(
            self.index,
            velodyne_path,
            semantic_label_path,
            self.calib_path,
            label_2_path
        )
        
        if success:
            print(f"Generated label_2 file: {label_2_path}")
        else:
            print(f"[WARN] Failed to generate label_2 file for frame {self.index}")
        
        return label_2_path

    # ------- Output helpers (paths) -------
    def get_label_path(self) -> str:
        # Keep template compatibility; not used for SemanticKITTI.
        return os.path.join(self.dataset_dir, "labels", f"{self.index_str}.label")


class SemanticKITTIDetectionOutputDataFrame(SemanticKITTIDetectionDataFrame):
    """
    Output dataframe for SemanticKITTI (pc-only)
    Creates standard folders under output_dir.
    """

    def __init__(self, frame_id: int, dataset_config: dict):
        # ensure absolute output_dir
        if "output_dir" in dataset_config:
            if not os.path.isabs(dataset_config["output_dir"]):
                dataset_config["output_dir"] = os.path.join(get_project_root_dir(), dataset_config["output_dir"])
            self.output_dir = dataset_config["output_dir"]
        else:
            raise ValueError("output_dir is required for SemanticKITTIDetectionOutputDataFrame")

        super().__init__(frame_id, dataset_config)
        self.init_dir()

        # payload fields
        self.point_cloud = None
        self.semantic_label_3d = None

    def init_dir(self):
        # setup output directories
        # point cloud
        dirname_pc = self.dataset_config.get("point_cloud_dirname", "velodyne")
        self.point_cloud_dir = os.path.join(self.output_dir, dirname_pc)
        # semantic labels
        dirname_sem3d = self.dataset_config.get("semantic_label_3d", "semantic_label_3d")
        self.semantic_label_dir = os.path.join(self.output_dir, dirname_sem3d)

        # Optional extras for output layout parity
        if "road_split_label" in self.dataset_config:
            self.road_split_label_dir = os.path.join(self.output_dir, self.dataset_config["road_split_label"])
        if "road_split_log" in self.dataset_config:
            self.road_split_log_dir = os.path.join(self.output_dir, self.dataset_config["road_split_log"])
        if "lidar_scene_info" in self.dataset_config:
            self.lidar_scene_info_dir = os.path.join(self.output_dir, self.dataset_config["lidar_scene_info"])

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def init_save_dir(self):
        os.makedirs(self.point_cloud_dir, exist_ok=True)
        os.makedirs(self.semantic_label_dir, exist_ok=True)

    def save(self):
        self.init_save_dir()
        if self.point_cloud is not None:
            self.save_point_cloud(self.point_cloud)
        if self.semantic_label_3d is not None:
            self.save_semantic_label_3d(self.semantic_label_3d)

    def save_point_cloud(self, point_cloud: np.ndarray):
        point_cloud_filepath = os.path.join(self.point_cloud_dir, f"{self.index_str}.bin")
        pc_utils.save_point_cloud(point_cloud_filepath, point_cloud)

    def save_semantic_label_3d(self, semantic_label_3d: np.ndarray):
        semantic_label_3d_filepath = os.path.join(self.semantic_label_dir, f"{self.index_str}.label")
        semantic_label_3d.astype(np.uint32).tofile(semantic_label_3d_filepath)
        print(f"Saved 3D semantic label to: {semantic_label_3d_filepath}")
