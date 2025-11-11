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
    Factory to instantiate the appropriate logic-scene generator.

    Args:
        dataset_type: Dataset identifier (e.g. "SemanticKITTI").
        dataset: Dataset instance bound to the generator.

    Returns:
        Configured logic-scene generator instance.
    """
    generators = {
        "SemanticKITTI": SemanticKITTILogicSceneInfoGenerator,
    }
    
    if dataset_type not in generators:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    generator_class = generators[dataset_type]
    
    return generator_class(dataset)

__all__ = ["create_logic_scene_generator", "BaseLogicSceneInfoGenerator", "SemanticKITTILogicSceneInfoGenerator"]
