import json
import math
import os
import random
import numpy as np
import yaml
from drivence.utils import mesh_utils
from drivence.utils.math_utils import get_str_numbers, get_list_str_numbers
from drivence.utils.path_utils import get_project_root_dir


class ObjectsLoader(object):
    """
    资产加载器，兼容 ShapeNetLoader 接口，用于加载 ShapeNet 格式的 3D 网格模型（如车辆、行人等）。

    该类负责从指定目录加载预定义类别的 3D 网格资产，支持模型路径映射、坐标对齐、尺度归一化、类别筛选等功能，
    为场景生成、点云仿真等模块提供标准化的网格加载接口。核心特性包括：
    1. 兼容 ShapeNetLoader 原有接口，支持平滑替换；
    2. 支持按类别、子类别筛选模型，适配多目标场景；
    3. 提供灵活的尺度归一化策略（固定比例/自动适配目标尺寸/帧特异性尺寸）；
    4. 支持 LiDAR/相机坐标系对齐，确保加载的模型与场景坐标系一致。

    配置依赖：
    需通过 config 参数传入 ShapeNet 相关配置，包含资产路径、模型文件名、类别定义、对齐参数、尺度参数等，
    具体配置项见 __init__ 方法中的参数解析。
    """

    def __init__(self, config: Dict):
        """
        初始化 ObjectsLoader，解析配置并构建模型路径映射、加载模型列表。

        Args:
            config (Dict): 全局配置字典，需包含 'shapenet' 键对应的子配置，关键配置项如下：
                - 'name': 资产名称标识（仅用于日志/标识，无实际功能）；
                - 'assets_path': 资产根目录的相对路径（基于项目根目录）；
                - 'object_file_name': 网格模型文件名（如 'model.obj'）；
                - 'object_dir_name': 模型文件所在的子目录名（如 'model'）；
                - 'strategy': 加载策略（预留字段，暂未使用）；
                - 'scale_ratio': 尺度比例（支持固定数值或 'auto'，'auto' 时自动适配目标尺寸）；
                - 'align_in_camera_coordinate': 相机坐标系对齐参数（字符串格式，如 "(0,0,0)"）；
                - 'align_in_lidar_coordinate': LiDAR 坐标系对齐参数列表（字符串格式，如 "[(0,0,0), (90,0,0)]"）；
                - 'target_lwh': 默认目标尺寸（长度、宽度、高度），格式为 (l, w, h)，用于 'auto' 尺度计算；
                - 'target_lwh_by_type' (可选): 按模型名称划分的目标尺寸字典，格式为 {model_name: (l, w, h)}；
                - 'frame_lwh_config' (可选): 按模型名称和帧号划分的目标尺寸字典，格式为 {model_name: {frame_num: (l, w, h)}}；
                - 'categories' (可选): 类别配置字典，格式为 {category: {subcategory: [model_name1, model_name2, ...]}}。
        """
        root_dir = get_project_root_dir()
        # Reuse ShapeNet configuration entries used by the main pipeline
        shapenet_config = config['shapenet']
        self.assets_name = shapenet_config['name']
        self.assets_dir = os.path.join(root_dir, shapenet_config['assets_path'])
        self.model_file_name = shapenet_config['object_file_name']
        self.model_dir_name = shapenet_config['object_dir_name']
        self.strategy = shapenet_config['strategy']
        self.scale_ratio = shapenet_config['scale_ratio']

        # Alignment configuration
        align_in_camera_coordinate = shapenet_config['align_in_camera_coordinate']
        align_in_lidar_coordinate = shapenet_config['align_in_lidar_coordinate']
        self.align_in_camera_coordinate = get_str_numbers(align_in_camera_coordinate)
        self.align_in_lidar_coordinate_list = get_list_str_numbers(align_in_lidar_coordinate)

        # Dimension configuration
        self.target_lwh = shapenet_config['target_lwh']
        self.target_lwh_by_type = shapenet_config.get('target_lwh_by_type', {})
        self.frame_lwh_config = shapenet_config.get('frame_lwh_config', {})

        # Category configuration
        self.categories = shapenet_config.get('categories', {})

        # Build the mapping from object name to asset path
        self.name_to_path_mapping = self._create_name_mapping()

        # Load the model list
        self.object_list = self._load_objects()

        # Backward compatibility: car_list mirrors object_list
        self.car_list = self.object_list

    def _load_objects(self) -> List[str]:
        """
        枚举所有可用的模型名称，过滤掉资产文件不存在的模型。

        遍历 categories 配置中的所有模型，检查对应的网格文件是否存在，仅保留存在的模型名称。

        Returns:
            List[str]: 可用模型名称列表（按类别配置遍历收集）。
        """
        object_list = []

        # Gather model names from the category configuration
        for category, subcategories in self.categories.items():
            for subcategory, models in subcategories.items():
                for model_name in models:
                    # Only keep models whose asset files exist
                    model_path = self._get_model_path_by_name(model_name)
                    if os.path.exists(model_path):
                        object_list.append(model_name)

        return object_list

    def _create_name_mapping(self) -> Dict[str, str]:
        """
        构建模型名称到资产相对路径的映射字典，用于快速查找模型文件路径。

        映射格式：{model_name: "category/subcategory/model_name"}，结合 assets_dir 可拼接为绝对路径。

        Returns:
            Dict[str, str]: 模型名称到相对路径的映射字典。
        """
        mapping = {}

        # 遍历类别配置，构建映射
        for category, subcategories in self.categories.items():
            for subcategory, models in subcategories.items():
                for model_name in models:
                    # 拼接相对路径（category/subcategory/model_name）
                    full_path = f"{category}/{subcategory}/{model_name}"
                    mapping[model_name] = full_path

        return mapping

    def _get_model_path_by_name(self, model_name: str) -> str:
        """
        根据模型名称获取网格文件的绝对路径。

        优先使用 _create_name_mapping 构建的映射查找路径，若未找到（如模型不在类别配置中），
        降级使用 legacy 目录结构（vehicles/cars/model_name/...）查找。

        Args:
            model_name (str): 模型名称（如 'car_001'）。

        Returns:
            str: 网格文件的绝对路径（格式：assets_dir/relative_path/model_dir_name/model_file_name）。
        """
        if model_name in self.name_to_path_mapping:
            full_path = self.name_to_path_mapping[model_name]
            return os.path.join(self.assets_dir, full_path, self.model_dir_name, self.model_file_name)
        else:
            # Fallback to legacy directory layout if the mapping is missing
            return os.path.join(self.assets_dir, "vehicles", "cars", model_name, self.model_dir_name,
                                self.model_file_name)
            # return os.path.join(self.assets_dir, "vehicles", "buses", model_name, self.model_dir_name, self.model_file_name)

    def get_mesh_path(self, car_name: str) -> str:
        """
        获取网格文件路径（兼容 ShapeNetLoader 接口）。

        直接调用 _get_model_path_by_name 方法，仅为适配原有接口命名习惯。

        Args:
            car_name (str): 模型名称（兼容原有接口的命名，实际支持任意模型类型）。

        Returns:
            str: 网格文件的绝对路径。
        """
        return self._get_model_path_by_name(car_name)

    def get_car_list(self, nums: Optional[int] = None) -> List[str]:
        """
        获取可用模型列表（兼容 ShapeNetLoader 接口）。

        支持返回全部模型或随机采样指定数量的模型，命名沿用 'car_list' 以保持兼容性。

        Args:
            nums (Optional[int], 可选): 需返回的模型数量，None 时返回全部模型。
                若 nums 大于可用模型总数，会抛出 ValueError（由 random.sample 触发）。

        Returns:
            List[str]: 模型名称列表（全部或随机采样后的子集）。
        """
        if nums is None:
            return self.car_list
        else:
            return random.sample(self.car_list, nums)

    def get_object_list(self, nums: Optional[int] = None) -> List[str]:
        """
        获取可用模型列表（新接口，推荐使用）。

        功能与 get_car_list 一致，命名更通用（支持非车辆类模型）。

        Args:
            nums (Optional[int], 可选): 需返回的模型数量，None 时返回全部模型。

        Returns:
            List[str]: 模型名称列表（全部或随机采样后的子集）。
        """
        if nums is None:
            return self.object_list
        else:
            return random.sample(self.object_list, nums)

    def load_mesh_by_name(self, car_name: str, scale_ratio: Optional[Union[float, str]] = None,
                          location: Optional[List[float]] = None, rotation: Optional[float] = None) -> object:
        """
        按模型名称加载网格，并执行对齐、尺度归一化、平移、旋转操作（兼容 ShapeNetLoader 接口）。

        核心流程：加载网格 → LiDAR 坐标系对齐 → 尺度归一化 → 平移 → 旋转，返回处理后的网格对象。

        Args:
            car_name (str): 模型名称（需在 object_list 中）。
            scale_ratio (Optional[Union[float, str]], 可选): 尺度比例，优先级：传入值 > 全局配置 > 自动计算。
                支持固定数值或 'auto'，'auto' 时根据 target_lwh 自动适配尺寸。
            location (Optional[List[float]], 可选): 平移坐标，格式为 [x, y, z]（单位：米），None 时不平移。
            rotation (Optional[float], 可选): 绕 Z 轴的旋转角度（单位：度），None 时不旋转。

        Returns:
            object: 处理后的网格模型对象（Open3D TriangleMesh 或其他兼容格式，需支持 scale/translate/rotate 方法）。

        Raises:
            FileNotFoundError: 若模型文件不存在（路径由 get_mesh_path 生成）；
            ValueError: 若 scale_ratio 为无效值，或 target_lwh 格式错误；
            RuntimeError: 若网格加载、对齐、归一化过程中出错。
        """
        # 获取网格路径并加载原始网格
        mesh_path = self.get_mesh_path(car_name)
        mesh_obj = mesh_utils.load_mesh(mesh_path)
        mesh_obj = self._align_mesh_in_lidar(mesh_obj)

        # Resolve scaling
        scale_ratio = self._get_scale_ratio_with_name(mesh_obj, scale_ratio, car_name)
        mesh_obj = self._normalize_mesh(mesh_obj, scale_ratio=scale_ratio, location=location, rotation=rotation)
        return mesh_obj

    def get_scale_ratio(self, car_name: str) -> float:
        """
        获取模型的尺度比例（兼容 ShapeNetLoader 接口）。

        加载模型并对齐后，根据全局配置的 scale_ratio 计算最终尺度比例（支持 'auto' 自动适配）。

        Args:
            car_name (str): 模型名称。

        Returns:
            float: 计算后的尺度比例（数值型）。
        """
        mesh_path = self.get_mesh_path(car_name)
        mesh_obj = mesh_utils.load_mesh(mesh_path)
        mesh_obj = self._align_mesh_in_lidar(mesh_obj)
        scale_ratio = self._get_scale_ratio_with_name(mesh_obj, self.scale_ratio, car_name)
        return scale_ratio

    def get_scale_ratio_by_frame(self, car_name: str, frame_number: int) -> float:
        """
        获取帧特异性的模型尺度比例（兼容 ShapeNetLoader 接口）。

        优先使用 frame_lwh_config 中该模型+帧号对应的目标尺寸，其次是 target_lwh_by_type，最后是默认 target_lwh。

        Args:
            car_name (str): 模型名称。
            frame_number (int): 帧号（用于匹配 frame_lwh_config 中的配置）。

        Returns:
            float: 帧特异性的尺度比例（数值型）。
        """
        mesh_path = self.get_mesh_path(car_name)
        mesh_obj = mesh_utils.load_mesh(mesh_path)
        mesh_obj = self._align_mesh_in_lidar(mesh_obj)
        scale_ratio = self._get_scale_ratio_with_name_and_frame(mesh_obj, self.scale_ratio, car_name, frame_number)
        return scale_ratio

    def _get_scale_ratio_with_name(self, mesh_obj: object, scale_ratio: Optional[Union[float, str]],
                                   car_name: str) -> float:
        """
        基于模型名称计算尺度比例（支持固定值和 'auto' 自动适配）。

        优先级：传入的 scale_ratio > 全局 scale_ratio；'auto' 时优先使用 target_lwh_by_type，其次是默认 target_lwh。

        Args:
            mesh_obj (object): 对齐后的网格模型对象。
            scale_ratio (Optional[Union[float, str]]): 传入的尺度比例（可能为 None）。
            car_name (str): 模型名称。

        Returns:
            float: 计算后的尺度比例（数值型）。

        Raises:
            ValueError: 若 scale_ratio 为无效字符串（非 'auto'），或 target_lwh 格式错误。
        """
        # 解析尺度比例（优先级：传入值 > 全局配置）
        if scale_ratio is None:
            scale_ratio = self.scale_ratio
        if scale_ratio == "auto":
            # Prefer type-specific target dimensions when defined
            if car_name in self.target_lwh_by_type:
                target_lwh = self.target_lwh_by_type[car_name]
                scale_ratio = mesh_utils.get_mesh_scale_ratio(mesh_obj, target_lwh)

            else:
                # Fall back to the default target dimensions
                scale_ratio = mesh_utils.get_mesh_scale_ratio(mesh_obj, self.target_lwh)
        return scale_ratio

    def _get_scale_ratio_with_name_and_frame(self, mesh_obj: object, scale_ratio: Optional[Union[float, str]],
                                             car_name: str, frame_number: int) -> float:
        """
        基于模型名称和帧号计算尺度比例（支持帧特异性尺寸配置）。

        优先级：frame_lwh_config（模型+帧号）> target_lwh_by_type（模型）> target_lwh（默认）。

        Args:
            mesh_obj (object): 对齐后的网格模型对象。
            scale_ratio (Optional[Union[float, str]]): 传入的尺度比例（可能为 None）。
            car_name (str): 模型名称。
            frame_number (int): 帧号。

        Returns:
            float: 计算后的尺度比例（数值型）。
        """
        # 解析尺度比例（优先级：传入值 > 全局配置）
        if scale_ratio is None:
            scale_ratio = self.scale_ratio
        if scale_ratio == "auto":
            # Prefer frame-specific dimensions when available
            frame_lwh = self.get_frame_lwh(car_name, frame_number)
            if frame_lwh is not None:
                scale_ratio = mesh_utils.get_mesh_scale_ratio(mesh_obj, frame_lwh)
            elif car_name in self.target_lwh_by_type:
                # Fall back to type-specific dimensions
                target_lwh = self.target_lwh_by_type[car_name]
                scale_ratio = mesh_utils.get_mesh_scale_ratio(mesh_obj, target_lwh)
            else:
                # Ultimately fall back to default dimensions
                scale_ratio = mesh_utils.get_mesh_scale_ratio(mesh_obj, self.target_lwh)
        return scale_ratio

    def get_frame_lwh(self, car_name: str, frame_number: int):
        """
        获取指定模型和帧号对应的目标尺寸（l, w, h）。

        从 frame_lwh_config 中查找，未找到则返回 None。

        Args:
            car_name (str): 模型名称。
            frame_number (int): 帧号。

        Returns:
            Optional[Tuple[float, float, float]]: 目标尺寸（l, w, h），未找到则返回 None。
        """
        if car_name in self.frame_lwh_config:
            frame_config = self.frame_lwh_config[car_name]
            if frame_number in frame_config:
                return frame_config[frame_number]
            return None
        return None

    def _align_mesh_in_lidar(self, mesh_obj: object) -> object:
        """
        将网格模型与 LiDAR 坐标系对齐。

        遍历 align_in_lidar_coordinate_list 中的所有旋转参数，依次对网格执行旋转操作（基于 XYZ 欧拉角）。

        Args:
            mesh_obj (object): 原始网格模型对象（需支持 get_rotation_matrix_from_xyz 和 rotate 方法）。

        Returns:
            object: 对齐后的网格模型对象。

        Raises:
            RuntimeError: 若网格不支持旋转相关方法，或旋转参数格式错误。
        """
        for align_in_lidar_coordinate in self.align_in_lidar_coordinate_list:
            # 根据 XYZ 欧拉角生成旋转矩阵（注意：角度单位需与 mesh_obj 要求一致，通常为弧度）
            R = mesh_obj.get_rotation_matrix_from_xyz(align_in_lidar_coordinate)
            mesh_obj.rotate(R)
        return mesh_obj

    def _normalize_mesh(self, mesh_obj: object, scale_ratio: float = 1.0,
                        location: Optional[List[float]] = None, rotation: Optional[float] = None) -> object:
        """
        对网格模型执行标准化操作：尺度缩放 → 平移 → 旋转（顺序固定）。

        Args:
            mesh_obj (object): 对齐后的网格模型对象。
            scale_ratio (float, 可选): 尺度比例，默认 1.0（不缩放）。
            location (Optional[List[float]], 可选): 平移坐标 [x, y, z]，默认 None（不平移）。
            rotation (Optional[float], 可选): 绕 Z 轴旋转角度（单位：度），默认 None（不旋转）。

        Returns:
            object: 标准化后的网格模型对象。

        Raises:
            ValueError: 若 location 维度不为 3，或 rotation 为无效角度值；
            RuntimeError: 若网格不支持 scale/translate/rotate 方法。
        """
        # 第一步：尺度缩放（以网格中心为缩放中心）
        if scale_ratio != 1.0:
            mesh_obj.scale(scale=scale_ratio, center=mesh_obj.get_center())

            # Then translate
        if location is not None:
            mesh_translation_vector = mesh_utils._get_translation_vector(mesh_obj, location)
            mesh_obj.translate(mesh_translation_vector)

            # Finally rotate
        if rotation is not None:
            import math
            rz_radians = math.radians(rotation)
            RZ = mesh_obj.get_rotation_matrix_from_xyz((0, 0, -rz_radians))
            mesh_obj.rotate(RZ)

        return mesh_obj

    # 类别相关辅助方法
    def get_models_by_category(self, category: str, subcategory: Optional[str] = None) -> List[str]:
        """
        按类别和子类别获取模型名称列表。

        Args:
            category (str): 主类别名称（需在 self.categories 中）。
            subcategory (Optional[str], 可选): 子类别名称，None 时返回该主类别下所有子类别的模型。

        Returns:
            List[str]: 匹配的模型名称列表，若类别不存在则返回空列表。
        """
        if category not in self.categories:
            return []

        if subcategory is None:
            # Collect every model under the specified category
            models = []
            for sub, model_list in self.categories[category].items():
                models.extend(model_list)
            return models
        else:
            # Return models for the requested subcategory
            return self.categories[category].get(subcategory, [])

    def get_all_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """
        获取完整的类别配置字典。

        Returns:
            Dict[str, Dict[str, List[str]]]: 类别配置，格式为 {category: {subcategory: [model_name1, ...]}}。
        """
        return self.categories