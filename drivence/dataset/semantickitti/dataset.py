import os
import numpy as np

from drivence.dataset.common.dataset_template import DataFrameTemplate
from drivence.dataset.common.point_cloud import PointCloud
from drivence.utils import pc_utils
from drivence.utils.path_utils import get_project_root_dir


class SemanticKITTIDetectionDataFrame(DataFrameTemplate):
    """
    SemanticKITTI 数据集的数据帧类，继承自 DataFrameTemplate，专注于激光雷达点云及语义标签的加载与管理。

    该类适配 SemanticKITTI 数据集的目录结构，提供点云、3D 语义标签、KITTI 格式 label_2 标注等核心数据的访问接口，
    支持从原始数据集加载数据，或自动从语义标签生成 KITTI 格式的目标检测标注（label_2 文件），
    为激光雷达目标检测、语义分割等任务提供标准化的数据封装。

    数据集目录结构示例（split 配置为 "sequences/08"）：
    root_dir/
      sequences/08/
        velodyne/          # 激光雷达点云文件（.bin）
          000000.bin
          ...
        labels/            # 3D 语义标签文件（.label）
          000000.label
          ...
        calib.txt          # 传感器标定文件（可选，用于生成 label_2）
        label_2/           # KITTI 格式目标检测标注（自动生成或预存）
          000000.txt
          ...
    """

    def __init__(self, frame_id: int, dataset_config: dict):
        """
        初始化 SemanticKITTIDetectionDataFrame，调用父类构造函数并初始化数据集配置。

        Args:
            frame_id (int): 数据帧的索引ID（如 0 对应文件 000000.bin/label）。
            dataset_config (dict): 数据集配置字典，关键配置项如下：
                - 'dataset_dir': 数据集根目录（绝对路径）；
                - 'split': 数据集分割（如 "sequences/08"），拼接在 dataset_dir 后形成实际数据目录；
                - 'point_cloud_dirname' (可选): 点云目录名，默认 "velodyne"；
                - 'semantic_label_3d' (可选): 3D 语义标签目录名，默认 "labels"；
                - 'road_split_label' (可选): 道路分割标签目录名（用于场景分割）；
                - 'road_split_log' (可选): 道路分割日志目录名；
                - 'lidar_scene_info' (可选): 激光雷达场景信息目录名；
                - 'label_2' (可选): KITTI 格式 label_2 目录名，默认 "label_2"。
        """
        super().__init__(frame_id, dataset_config)

    def init_dir(self):
        """
        初始化数据集各目录路径，根据配置字典拼接点云、语义标签、标注文件等的存储路径。

        核心逻辑：
        1. 点云目录：优先使用配置中的 'point_cloud_dirname'，默认 "velodyne"；
        2. 3D 语义标签目录：优先使用配置中的 'semantic_label_3d'，默认 "labels"；
        3. 可选目录：道路分割标签、场景信息等目录（按需配置）；
        4. label_2 目录：优先使用配置中的 'label_2'，默认 "label_2"；
        5. 标定文件：优先查找 dataset_dir 下的 calib.txt，其次查找 calib 子目录下的 calib.txt；
        6. 未使用目录：SemanticKITTI 仅关注点云相关数据，设置图像、深度等目录为 None。
        """
        # 激光雷达点云目录（兼容配置中的自定义目录名）
        if "point_cloud_dirname" in self.dataset_config:
            self.point_cloud_dir = os.path.join(self.dataset_dir, self.dataset_config["point_cloud_dirname"])
        else:
            self.point_cloud_dir = os.path.join(self.dataset_dir, "velodyne")

        # 3D 语义标签目录（.label 文件），兼容配置中的自定义目录名
        sem3d_dirname = self.dataset_config.get("semantic_label_3d", "labels")
        self.semantic_label_dir = os.path.join(self.dataset_dir, sem3d_dirname)

        # 可选目录（按需配置，用于场景分割、日志存储等）
        if "road_split_label" in self.dataset_config:
            self.road_split_label_dir = os.path.join(self.dataset_dir, self.dataset_config["road_split_label"])
        if "road_split_log" in self.dataset_config:
            self.road_split_log_dir = os.path.join(self.dataset_dir, self.dataset_config["road_split_log"])
        if "lidar_scene_info" in self.dataset_config:
            self.lidar_scene_info_dir = os.path.join(self.dataset_dir, self.dataset_config["lidar_scene_info"])

        # KITTI 格式 label_2 目录（用于目标检测标注）
        if "label_2" in self.dataset_config:
            self.label_2_dir = os.path.join(self.dataset_dir, self.dataset_config["label_2"])
        else:
            self.label_2_dir = os.path.join(self.dataset_dir, "label_2")

        # 标定文件路径（用于生成 label_2 时的坐标转换）
        calib_path = os.path.join(self.dataset_dir, "calib.txt")
        if not os.path.exists(calib_path):
            calib_path = os.path.join(self.dataset_dir, "calib", "calib.txt")
        self.calib_path = calib_path if os.path.exists(calib_path) else None

        # SemanticKITTI 仅处理点云数据，设置图像、深度、2D 标签等目录为 None
        self.image_dir = None
        self.calib_dir = None
        self.depth_dir = None
        self.label_dir = None

    # ------- 点云相关核心接口 -------
    def get_point_cloud(self) -> PointCloud:
        """
        加载当前帧的激光雷达点云，返回封装后的 PointCloud 对象（含坐标和反射率）。

        SemanticKITTI 点云文件为 .bin 格式，每个点包含 4 维信息（X, Y, Z, 反射率），
        该方法加载后分离坐标（前 3 维）和反射率（第 4 维），封装为 PointCloud 结构。

        Returns:
            PointCloud: 封装后的点云对象，包含坐标（.point_cloud）和反射率（.reflection）属性。

        Raises:
            FileNotFoundError: 若点云文件不存在（路径：point_cloud_dir/{index_str}.bin）；
            RuntimeError: 若点云加载失败（如文件损坏、格式错误）。
        """
        # 拼接点云文件路径（index_str 为帧ID的 6 位字符串，如 0→"000000"）
        point_cloud_filepath = os.path.join(self.point_cloud_dir, f"{self.index_str}.bin")
        # 加载点云（4 维：X, Y, Z, 反射率）
        point_cloud_npy = pc_utils.load_pc(point_cloud_filepath, demension=4)
        # 封装为 PointCloud 对象
        pc = PointCloud()
        pc.set_point_cloud(point_cloud_npy[:, :3])  # 坐标（X, Y, Z）
        pc.set_reflection(point_cloud_npy[:, 3])  # 反射率
        return pc

    def get_semantic_label_3d(self) -> Optional[np.ndarray]:
        """
        加载当前帧的 3D 语义标签（SemanticKITTI 原生 .label 文件）。

        语义标签文件为 uint32 格式，每个值对应一个点的语义类别（需调用者根据需求重映射标签），
        若文件不存在，打印警告并返回 None。

        Returns:
            Optional[np.ndarray]: 语义标签数组（形状为 (N,)，N 为点云点数， dtype=np.uint32），
                文件不存在时返回 None。

        Raises:
            RuntimeError: 若标签文件读取失败（如文件损坏、格式错误）。
        """
        # 拼接语义标签文件路径
        label_path = os.path.join(self.semantic_label_dir, f"{self.index_str}.label")
        if not os.path.exists(label_path):
            print(f"Warning: semantic label file not found: {label_path}")
            return None
        labels = np.fromfile(label_path, dtype=np.uint32)
        return labels

    # ------- 未使用接口（点云专用数据集，返回 None） -------
    def get_image(self):
        """SemanticKITTI 无图像数据，返回 None（兼容父类接口）。"""
        return None

    def get_depth(self):
        """SemanticKITTI 无深度数据，返回 None（兼容父类接口）。"""
        return None

    def get_calibration(self):
        """标定数据通过 self.calib_path 访问，此接口返回 None（兼容父类接口）。"""
        return None

    def get_label(self):
        """SemanticKITTI 基线无预定义的 2D/3D 检测标签，返回 None（兼容父类接口）。"""
        # No 2D/3D detection labels in SemanticKITTI baseline
        return None

    def get_label_2_path(self) -> str:
        """
        获取 KITTI 格式 label_2 标注文件的路径（用于目标检测）。

        Returns:
            str: label_2 文件路径（格式：label_2_dir/{index_str}.txt）。
        """
        return os.path.join(self.label_2_dir, f"{self.index_str}.txt")

    def get_label_2(self) -> str:
        """
        获取或自动生成 KITTI 格式的 label_2 标注文件（从 3D 语义标签生成）。

        核心逻辑：
        1. 若 label_2 文件已存在，直接返回文件路径；
        2. 若文件不存在，使用 SemanticKITTILabel2Generator 从语义标签生成：
           a. 检查语义标签文件（.label 或 .npy 格式）是否存在；
           b. 检查标定文件（calib.txt）是否存在；
           c. 生成 label_2 文件并保存到 label_2_dir；
        3. 生成成功返回文件路径，失败返回路径并打印警告。

        Returns:
            str: label_2 文件路径（无论生成成功与否，均返回路径）。

        Raises:
            ImportError: 若 SemanticKITTILabel2Generator 未找到；
            RuntimeError: 若生成过程中出现文件读取/写入异常。
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

    # ------- 输出路径辅助接口 -------
    def get_label_path(self) -> str:
        """
        获取标签文件路径（兼容父类接口，SemanticKITTI 中未实际使用）。

        Returns:
            str: 标签文件路径（格式：dataset_dir/labels/{index_str}.label）。
        """
        # Keep template compatibility; not used for SemanticKITTI.
        return os.path.join(self.dataset_dir, "labels", f"{self.index_str}.label")


class SemanticKITTIDetectionOutputDataFrame(SemanticKITTIDetectionDataFrame):
    """
    SemanticKITTI 输出数据帧类，继承自 SemanticKITTIDetectionDataFrame，专注于处理生成数据的保存。

    该类扩展了父类的功能，支持将生成的点云、3D 语义标签等数据保存到指定的输出目录，
    自动创建标准输出目录结构，确保输出数据与原始 SemanticKITTI 格式兼容，
    适用于数据增强、场景生成等需要保存结果的任务。
    """

    def __init__(self, frame_id: int, dataset_config: dict):
        """
        初始化输出数据帧，校验并处理输出目录配置，调用父类构造函数。

        Args:
            frame_id (int): 数据帧的索引ID。
            dataset_config (dict): 数据集配置字典，必须包含 'output_dir' 键（输出目录路径）。

        Raises:
            ValueError: 若 dataset_config 中未包含 'output_dir'。
        """
        # 校验并处理输出目录（确保为绝对路径）
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
        """
        初始化输出目录结构，根据配置拼接点云、语义标签等输出路径。

        核心逻辑：
        1. 点云输出目录：优先使用配置中的 'point_cloud_dirname'，默认 "velodyne"；
        2. 3D 语义标签输出目录：优先使用配置中的 'semantic_label_3d'，默认 "semantic_label_3d"；
        3. 可选输出目录：道路分割标签、场景信息等（按需配置，与输入目录结构对齐）。
        """
        # 初始化输出目录（点云、语义标签等）
        # 点云输出目录
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
        """
        批量设置待保存的数据字段（如点云、语义标签）。

        支持通过关键字参数动态设置实例属性，例如：
        df.set(point_cloud=pc_npy, semantic_label_3d=sem_labels)

        Args:
            **kwargs: 关键字参数，键为属性名（如 'point_cloud'），值为待保存的数据。
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def init_save_dir(self):
        """创建输出目录（含所有子目录），确保保存时目录存在（已存在则忽略）。"""
        os.makedirs(self.point_cloud_dir, exist_ok=True)
        os.makedirs(self.semantic_label_dir, exist_ok=True)

    def save(self):
        """
        保存待输出的数据（点云和 3D 语义标签），自动先初始化输出目录。

        仅保存已设置的字段（self.point_cloud 或 self.semantic_label_3d 不为 None 时才保存）。
        """
        self.init_save_dir()
        if self.point_cloud is not None:
            self.save_point_cloud(self.point_cloud)
        if self.semantic_label_3d is not None:
            self.save_semantic_label_3d(self.semantic_label_3d)

    def save_point_cloud(self, point_cloud: np.ndarray):
        """
        保存点云数据到输出目录（SemanticKITTI 兼容的 .bin 格式）。

        Args:
            point_cloud (np.ndarray): 点云数组，形状为 (N, 3) 或 (N, 4)（含反射率）， dtype 为 float32。

        Raises:
            ValueError: 若点云数组维度非法（非 3 或 4 维）；
            RuntimeError: 若文件写入失败（如权限不足、磁盘空间不足）。
        """
        # 拼接点云输出路径
        point_cloud_filepath = os.path.join(self.point_cloud_dir, f"{self.index_str}.bin")
        # 保存点云（pc_utils.save_point_cloud 需支持 .bin 格式写入）
        pc_utils.save_point_cloud(point_cloud_filepath, point_cloud)

    def save_semantic_label_3d(self, semantic_label_3d: np.ndarray):
        """
        保存 3D 语义标签到输出目录（SemanticKITTI 兼容的 .label 格式，uint32 类型）。

        Args:
            semantic_label_3d (np.ndarray): 语义标签数组，形状为 (N,)（N 为点云点数）。

        Raises:
            ValueError: 若标签数组维度非法（非 1 维）；
            RuntimeError: 若文件写入失败。
        """
        # 拼接标签输出路径
        semantic_label_3d_filepath = os.path.join(self.semantic_label_dir, f"{self.index_str}.label")
        # 转换为 uint32 格式并写入文件（SemanticKITTI 原生标签格式）
        semantic_label_3d.astype(np.uint32).tofile(semantic_label_3d_filepath)
        print(f"Saved 3D semantic label to: {semantic_label_3d_filepath}")
