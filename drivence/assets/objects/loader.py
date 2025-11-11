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
    """Asset loader compatible with the ShapeNetLoader interface."""
    
    def __init__(self, config):
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
    
    def _load_objects(self):
        """Enumerate all available models."""
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
    
    def _create_name_mapping(self):
        """Create a mapping between model names and asset sub-paths."""
        mapping = {}
        
        # Build the mapping for every model
        for category, subcategories in self.categories.items():
            for subcategory, models in subcategories.items():
                for model_name in models:
                    # Compose the relative path
                    full_path = f"{category}/{subcategory}/{model_name}"
                    mapping[model_name] = full_path
        
        return mapping
    
    def _get_model_path_by_name(self, model_name):
        """Return the on-disk path for the specified model."""
        if model_name in self.name_to_path_mapping:
            full_path = self.name_to_path_mapping[model_name]
            return os.path.join(self.assets_dir, full_path, self.model_dir_name, self.model_file_name)
        else:
            # Fallback to legacy directory layout if the mapping is missing
            return os.path.join(self.assets_dir, "vehicles", "cars", model_name, self.model_dir_name, self.model_file_name)
            # return os.path.join(self.assets_dir, "vehicles", "buses", model_name, self.model_dir_name, self.model_file_name)
    
    def get_mesh_path(self, car_name):
        """Return the mesh path (ShapeNetLoader-compatible)."""
        return self._get_model_path_by_name(car_name)
    
    def get_car_list(self, nums=None):
        """Return the list of available models (ShapeNetLoader-compatible)."""
        if nums is None:
            return self.car_list
        else:
            return random.sample(self.car_list, nums)
    
    def get_object_list(self, nums=None):
        """Return the list of available models (new interface)."""
        if nums is None:
            return self.object_list
        else:
            return random.sample(self.object_list, nums)
    
    def load_mesh_by_name(self, car_name, scale_ratio=None, location=None, rotation=None):
        """Load and normalize a mesh (ShapeNetLoader-compatible)."""
        mesh_path = self.get_mesh_path(car_name)
        mesh_obj = mesh_utils.load_mesh(mesh_path)
        mesh_obj = self._align_mesh_in_lidar(mesh_obj)
        
        # Resolve scaling
        scale_ratio = self._get_scale_ratio_with_name(mesh_obj, scale_ratio, car_name)
        mesh_obj = self._normalize_mesh(mesh_obj, scale_ratio=scale_ratio, location=location, rotation=rotation)
        return mesh_obj
    
    def get_scale_ratio(self, car_name):
        """Return the scale ratio for a model (ShapeNetLoader-compatible)."""
        mesh_path = self.get_mesh_path(car_name)
        mesh_obj = mesh_utils.load_mesh(mesh_path)
        mesh_obj = self._align_mesh_in_lidar(mesh_obj)
        scale_ratio = self._get_scale_ratio_with_name(mesh_obj, self.scale_ratio, car_name)
        return scale_ratio
    
    def get_scale_ratio_by_frame(self, car_name, frame_number):
        """Return the frame-specific scale ratio (ShapeNetLoader-compatible)."""
        mesh_path = self.get_mesh_path(car_name)
        mesh_obj = mesh_utils.load_mesh(mesh_path)
        mesh_obj = self._align_mesh_in_lidar(mesh_obj)
        scale_ratio = self._get_scale_ratio_with_name_and_frame(mesh_obj, self.scale_ratio, car_name, frame_number)
        return scale_ratio
    
    def _get_scale_ratio_with_name(self, mesh_obj, scale_ratio, car_name):
        """Compute a type-specific scale ratio based on the asset name."""
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
    
    def _get_scale_ratio_with_name_and_frame(self, mesh_obj, scale_ratio, car_name, frame_number):
        """Compute a type/frame-specific scale ratio based on the asset name and frame id."""
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
    
    def get_frame_lwh(self, car_name, frame_number):
        """Retrieve the LWH dimensions for a specific asset and frame."""
        if car_name in self.frame_lwh_config:
            frame_config = self.frame_lwh_config[car_name]
            if frame_number in frame_config:
                return frame_config[frame_number]
            return None
        return None
    
    def _align_mesh_in_lidar(self, mesh_obj):
        """Align a mesh with the LiDAR coordinate system."""
        for align_in_lidar_coordinate in self.align_in_lidar_coordinate_list:
            R = mesh_obj.get_rotation_matrix_from_xyz(align_in_lidar_coordinate)
            mesh_obj.rotate(R)
        return mesh_obj
    
    def _normalize_mesh(self, mesh_obj, scale_ratio=1.0, location=None, rotation=None):
        """Normalize a mesh using the same procedure as the ShapeNet loader."""
        # Scale first
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
    
    # Helper methods for category-based access
    def get_models_by_category(self, category, subcategory=None):
        """Return models grouped by category and optional subcategory."""
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
    
    def get_all_categories(self):
        """Return the entire category configuration."""
        return self.categories 