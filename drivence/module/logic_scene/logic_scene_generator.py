import os
from typing import List
from abc import ABC, abstractmethod

from drivence.dataset.common.dataset_template import DataFrameTemplate
from drivence.utils import path_utils
from drivence.entity.scene_info import SceneInfo, generate_scene_info, generate_vehicle_infoes


class BaseLogicSceneInfoGenerator(ABC):
    """Abstract base class for logic-scene info generators."""
    
    def __init__(self, dataset: DataFrameTemplate = None):
        self.dataset = dataset
        self.filename = "scene_{}.json"
    
    def get_dataset_type(self) -> str:
        """Return the dataset identifier supported by this generator."""
        pass
    
    def convert_pose_to_scene_infos(self, positions, rz_degrees, scale, obj_indexes, obj_names, assets_name, frame_id,
                                    save_path) -> SceneInfo:
        """Convert poses into a serialized SceneInfo payload."""
        v_info = generate_vehicle_infoes(obj_indexes, obj_names, positions, rz_degrees, scale)
        scene_info = generate_scene_info(self.get_dataset_type(),
                                         "",
                                         frame_id,
                                         v_info,
                                         assets_name)
        scene_info = SceneInfo(scene_info)
        path_utils.create_parent_dir(save_path)
        scene_info.to_json_with_vehicles(save_path=save_path)
        return scene_info


class SemanticKITTILogicSceneInfoGenerator(BaseLogicSceneInfoGenerator):
    """Logic scene generator tailored for SemanticKITTI."""

    def __init__(self, dataset: DataFrameTemplate):
        super().__init__(dataset)
        self.dataset = dataset

    def get_dataset_type(self) -> str:
        return "SemanticKITTI"

    def convert_trajectories_to_scene_infos(self, trajectories, car_names, assets_name, save_dir, scale_ratio_arr):
        print("Warning: Trajectory-to-scene conversion is not implemented for SemanticKITTI")
        return []


def create_logic_scene_generator(dataset_type: str, dataset: DataFrameTemplate = None):
    """
    逻辑场景生成器的工厂函数，根据数据集类型实例化对应的逻辑场景信息生成器。

    采用工厂模式封装不同数据集的逻辑场景生成逻辑，通过数据集类型（dataset_type）自动匹配对应的生成器类，
    屏蔽不同数据集的场景构造差异，为上层模块（如 LiDAR 数据生成流水线）提供统一的场景信息生成接口。
    生成器主要用于构造包含车辆、背景、语义标签等核心信息的逻辑场景，为后续点云仿真、融合计算提供基础。

    核心逻辑：
    1. 维护数据集类型与生成器类的映射关系（支持扩展新数据集）；
    2. 校验输入数据集类型的有效性，不支持则抛出异常；
    3. 根据映射关系获取对应的生成器类，传入数据集实例完成初始化；
    4. 返回配置好的逻辑场景生成器实例，供上层调用。

    Args:
        dataset_type (str): 数据集类型标识符（需与映射字典中的键完全匹配）。
            目前支持的数据集：
            - "SemanticKITTI": SemanticKITTI 数据集，对应 SemanticKITTILogicSceneInfoGenerator 生成器；
            扩展说明：新增数据集时，需在 generators 字典中添加 "数据集类型": 生成器类 的映射。
        dataset (DataFrameTemplate, 可选): 绑定到生成器的数据集实例，默认 None。
            该实例需遵循 DataFrameTemplate 抽象接口，提供生成器所需的数据集元信息（如场景配置、标签体系等）；
            若为 None，生成器将使用默认配置初始化（具体依赖生成器类的实现）。

    Returns:
        LogicSceneInfoGenerator: 配置完成的逻辑场景生成器实例，继承自基础抽象类 LogicSceneInfoGenerator。
            实例需实现场景信息构造的核心方法（如生成车辆列表、语义标签映射等），具体功能由数据集专属生成器定义。
    """
    generators = {
        "SemanticKITTI": SemanticKITTILogicSceneInfoGenerator,
    }
    
    if dataset_type not in generators:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    generator_class = generators[dataset_type]
    
    return generator_class(dataset)

__all__ = ["create_logic_scene_generator", "BaseLogicSceneInfoGenerator", "SemanticKITTILogicSceneInfoGenerator"]
