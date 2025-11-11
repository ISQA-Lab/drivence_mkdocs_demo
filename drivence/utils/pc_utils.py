from drivence.utils.format_convert import numpy_to_torch
import numpy as np
import open3d as o3d
from matplotlib import cm
from lidar_core.intensity_formula.Formula import lambert_beckmann_model, eq_7, data


def load_pc_xyzr(path: str):
    return load_pc(path, 4)


def load_pc(path: str, demension=4, format=None):
    if format is None:
        if path.split(".")[-1] == "npy":
            format = "npy"
        elif path.split(".")[-1] == "bin":
            format = "bin"
        else:
            raise ValueError(f"Unsupported file format: {path.split('.')[-1]}")
    return _load_pc_detail(path, demension, format)


def _load_pc_detail(path: str, demension, format):
    if format == "npy":
        example = np.load(path).astype(np.float32)
    else:
        example = np.fromfile(path, dtype=np.float32).reshape(-1, demension)
    return example


def save_point_cloud(file_path, point_cloud):
    point_cloud.astype(np.float32).tofile(file_path)  # Final point cloud must be stored as float32 to guarantee it can be reloaded


def rotate_points_along_z(points: np.ndarray, angle: np.ndarray, return_type: str = "numpy") -> np.ndarray:
    import torch
    """Rotate batched point clouds around the z-axis."""
    points = numpy_to_torch(points)
    angle = numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)

    if return_type == "numpy":
        return points_rot.numpy()
    else:
        return points_rot


def load_pcd(path: str, return_pcd=False):
    if path.split(".")[-1] == "npy":
        example = np.load(path).astype(np.float32)
    else:
        example = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    example_xyz = example[:, :3]

    if return_pcd:
        return example
    return example_xyz


def get_reflection_intensities_by_objects(intensity_config_dict: dict = None):
    """Load reflection intensity parameters for each model from the intensity configuration."""
    # Prefer the provided dictionary; otherwise, fall back to the YAML file for backward compatibility
    if intensity_config_dict is not None:
        intensity_config = intensity_config_dict
    else:
        # Fall back to the legacy config file path (requires the file to be present alongside the project)
        pass
    
    try:
        
        # Extract the intensity configuration and default parameters
        intensity_values = intensity_config['intensity_config']
        default_intensity_before = intensity_config.get('default_intensity_before', 300)
        default_physics = intensity_config.get('default_physics_params', {
            'p_lambda': 0.7,
            'kd': 0.95,
            'm': 0.15,
            'theta_T': 10
        })
        
        # Build a mapping from model name to configuration parameters
        model_config_mapping = {}
        
        # Iterate through categories and subcategories
        for category, subcategories in intensity_values.items():
            for subcategory, config in subcategories.items():
                if isinstance(config, dict) and 'models' in config:
                    # Retrieve physics parameters, falling back to defaults when absent
                    p_lambda = config.get('p_lambda', default_physics['p_lambda'])
                    kd = config.get('kd', default_physics['kd'])
                    m = config.get('m', default_physics['m'])
                    theta_T = config.get('theta_T', default_physics['theta_T'])
                    
                    models = config['models']
                    # Assign the same configuration to every model within this subcategory
                    for model_name in models:
                        model_config_mapping[model_name] = {
                            'p_lambda': p_lambda,
                            'kd': kd,
                            'm': m,
                            'theta_T': theta_T
                        }
                        
                else:
                    pass
        
        return model_config_mapping
        
    except Exception as e:
        pass
        return {}


def calculate_surface_normal_from_tangent_plane(point_xyz, object_points, k_neighbors=10):
    """Estimate a surface normal by fitting a tangent plane around the query point."""
    # If there are too few points, fall back to a default normal vector
    if len(object_points) < k_neighbors:
        return np.array([0, 0, 1]), 0.0
    
    # Compute the distance from the query point to every other point
    distances = np.linalg.norm(object_points - point_xyz, axis=1)
    
    # Select the nearest k neighbors (excluding the query point itself)
    nearest_indices = np.argsort(distances)[1:k_neighbors+1]
    nearest_points = object_points[nearest_indices]
    
    # Compute the centroid of the neighborhood
    centroid = np.mean(nearest_points, axis=0)
    
    # Build the covariance matrix
    centered_points = nearest_points - centroid
    covariance_matrix = np.dot(centered_points.T, centered_points)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # The eigenvector associated with the smallest eigenvalue is perpendicular to the fitted plane
    normal_vector = eigenvectors[:, 0]

    # Ensure the normal points toward the LiDAR (assumed to be at the origin)
    if np.dot(normal_vector, point_xyz) < 0:
        normal_vector = -normal_vector
    
    # Compute the incidence angle between the ray (origin -> point) and the normal vector
    laser_direction = point_xyz / np.linalg.norm(point_xyz)
    
    # Resolve the acute incidence angle (0–90 degrees)
    cos_theta = abs(np.dot(laser_direction, normal_vector))
    cos_theta = np.clip(cos_theta, 0, 1)
    theta_rad = np.arccos(cos_theta)
    
    # Convert to degrees
    theta_deg = np.degrees(theta_rad)
    
    return normal_vector, theta_deg


def calculate_physics_based_intensity(model_name, point_xyz, model_config, object_points=None):
    """Compute physics-based reflection intensity using the Eq.15 radiometric model."""
    # Retrieve LiDAR system parameters
    params = data()

    # Distance from LiDAR to point
    distance = np.linalg.norm(point_xyz)

    # Incidence angle
    normal_vector, theta = calculate_surface_normal_from_tangent_plane(point_xyz, object_points)


    # Physical parameters per object
    p_lambda = model_config.get('p_lambda', 0.7)
    kd = model_config.get('kd', 0.95)
    m = model_config.get('m', 0.15)
    theta_T = model_config.get('theta_T', 10)

    # System constant C_lambda derived from hardware parameters
    Pt_lambda_base = 70
    Dr = (params["Dr"][0] + params["Dr"][1]) / 2                    # Receiver aperture (100 mm)
    n_atm = (params["n_atm_lambda"][0] + params["n_atm_lambda"][1]) / 2  # Atmospheric transmission efficiency (≈0.9)
    n_sys = (params["n_sys_lambda"][0] + params["n_sys_lambda"][1]) / 2  # System efficiency (≈0.4)

    C_lambda = eq_7(Pt_lambda_base, Dr, n_atm, n_sys, R=distance, max_compensation_distance=25.0)

    # Reference incidence angle (0 degrees in the cited paper)
    theta_s = params.get("theta_s", 0)

    # Radiometric model combining Lambertian and Beckmann components; protect against zero distance
    if distance <= 1e-6:
        distance = 1e-6
    final_intensity = lambert_beckmann_model(
        theta=theta,
        kd=kd,
        m=m,
        theta_T=theta_T,
        C_lambda=C_lambda,
        p_lambda=p_lambda,
        R=distance,
    )

    # i_max controls the normalization range for the specific LiDAR sensor
    # i_max = eq_9(C_lambda, 1, 10)
    # i_max = eq_9(C_lambda, p_lambda, distance)
    i_max = 85 # TODO: confirm whether this value is specific to SemanticKITTI

    normalized_intensity = np.clip(final_intensity / i_max, 0.0, 1.0)

    return normalized_intensity


def pc_numpy_2_pcr_with_instance_intensity(mixed_pc_three_dims, instance_mask, obj_names=None, model_intensity_mapping=None, object_point_clouds=None) -> np.ndarray:
    """Attach intensity values to a mixed point cloud using instance segmentation metadata."""
    # Ensure each point has three spatial dimensions
    assert mixed_pc_three_dims.shape[1] == 3
    # Total number of points in the mixed point cloud
    point_nums = mixed_pc_three_dims.shape[0]
 
    # Allocate an intensity value for every point based on its instance id
    intensity_values = np.zeros(point_nums)
 
    if obj_names is not None and len(obj_names) > 0 and model_intensity_mapping is not None:
        # Use the supplied configuration to determine intensities per model

        # Track models that fall back to defaults to avoid repeated warnings
        warned_models = set()

        # Keep a record of computed intensities for each object instance
        object_intensity_info = {}
 
        for point_idx, instance_idx in enumerate(instance_mask):
            if 0 <= instance_idx < len(obj_names):
                model_name = obj_names[instance_idx]
 
                # Use the instance index to build a unique key, consistent with create_fused_pointcloud_with_intensity
                unique_key = f"{model_name}_instance{instance_idx}"
 
                if model_name in model_intensity_mapping:
                    model_config = model_intensity_mapping[model_name]
 
                    # Current point coordinates
                    point_xyz = mixed_pc_three_dims[point_idx]
 
                    # Physics-based intensity; prefer per-instance data when available
                    object_points = None
                    if object_point_clouds:
                        object_points = object_point_clouds.get(unique_key)
                        if object_points is None:
                            object_points = object_point_clouds.get(model_name)
                    
                    final_intensity = calculate_physics_based_intensity(model_name, point_xyz, model_config, object_points)
                    
                    intensity_values[point_idx] = final_intensity

                    # Cache the intensity result for this instance
                    if unique_key not in object_intensity_info:
                        object_intensity_info[unique_key] = final_intensity
                else:
                    # TODO: consider making the default model configurable
                    model_config = model_intensity_mapping["Car_1"]
                    # model_config = model_intensity_mapping["Bus"]
 
                    # Current point coordinates
                    point_xyz = mixed_pc_three_dims[point_idx]
 
                    # Physics-based intensity; prefer per-instance data when available
                    object_points = None
                    if object_point_clouds:
                        object_points = object_point_clouds.get(unique_key)
                        if object_points is None:
                            object_points = object_point_clouds.get(model_name)
                    
                    final_intensity = calculate_physics_based_intensity(model_name, point_xyz, model_config,
                                                                        object_points)

                    # Cache the intensity result for this instance
                    if unique_key not in object_intensity_info:
                        object_intensity_info[unique_key] = final_intensity
            else:
                intensity_values[point_idx] = 0.5  # Default fallback when instance id is invalid
 
        # Intensity details are left silent intentionally
    else:
        # Without configuration data, fall back to a uniform default value
        intensity_values[:] = 0.5
 
    # Append the intensity column to obtain an (N, 4) array
    intensity_column = intensity_values.reshape(-1, 1)
    mixed_pc = np.concatenate([mixed_pc_three_dims, intensity_column], axis=1)
    
    return mixed_pc


def compare_intensity_statistics(pc_bg_intensity, obj_names, instance_mask, obj_intensity, pc_bg_remain, pc_obj_remain):
    """Compute descriptive statistics for inserted-object intensities."""
    statistics = {}
 
    # Per-object intensity statistics
    if obj_names is not None and len(obj_names) > 0:
        # Group by object name so identical assets are aggregated
        obj_name_groups = {}
 
        for i, obj_name in enumerate(obj_names):
            # Points belonging to this instance
            obj_point_indices = np.where(instance_mask == i)[0]
 
            if len(obj_point_indices) > 0:
                # Intensities for the current instance
                current_obj_intensity = obj_intensity[obj_point_indices]
 
                # Merge with previous entries for the same asset
                if obj_name in obj_name_groups:
                    obj_name_groups[obj_name].extend(current_obj_intensity)
                else:
                    obj_name_groups[obj_name] = list(current_obj_intensity)
 
        # Compute statistics for each asset type
        for obj_name, all_intensities in obj_name_groups.items():
            all_intensities = np.array(all_intensities)
 
            # Summary statistics
            obj_intensity_mean = np.mean(all_intensities)
            obj_intensity_std = np.std(all_intensities)
            obj_intensity_min = np.min(all_intensities)
            obj_intensity_max = np.max(all_intensities)
            obj_intensity_median = np.median(all_intensities)
            obj_intensity_q25 = np.percentile(all_intensities, 25)
            obj_intensity_q75 = np.percentile(all_intensities, 75)
            obj_intensity_iqr = obj_intensity_q75 - obj_intensity_q25
            obj_intensity_cv = obj_intensity_std / obj_intensity_mean if obj_intensity_mean != 0 else 0
 
            # Store statistics
            statistics[obj_name] = {
                'count': len(all_intensities),
                'mean': obj_intensity_mean,
                'std': obj_intensity_std,
                'min': obj_intensity_min,
                'max': obj_intensity_max,
                'median': obj_intensity_median,
                'q25': obj_intensity_q25,
                'q75': obj_intensity_q75,
                'iqr': obj_intensity_iqr,
                'cv': obj_intensity_cv
            }
 
        # Mark entries that have no associated points
        for obj_name in obj_names:
            if obj_name not in obj_name_groups:
                statistics[obj_name] = {
                    'count': 0,
                    'warning': 'missing point cloud data'
                }
 
    # Aggregate statistics across all inserted objects
    if obj_names is not None and len(obj_names) > 0:
        all_obj_intensity = obj_intensity
        all_obj_mean = np.mean(all_obj_intensity)
        all_obj_std = np.std(all_obj_intensity)
 
        # Additional metrics for the overall distribution
        all_obj_median = np.median(all_obj_intensity)
        all_obj_q25 = np.percentile(all_obj_intensity, 25)
        all_obj_q75 = np.percentile(all_obj_intensity, 75)
        all_obj_iqr = all_obj_q75 - all_obj_q25
        all_obj_cv = all_obj_std / all_obj_mean if all_obj_mean != 0 else 0
 
        statistics['overall'] = {
            'inserted_count': len(all_obj_intensity),
            'inserted_mean': all_obj_mean,
            'inserted_std': all_obj_std,
            'inserted_median': all_obj_median,
            'inserted_q25': all_obj_q25,
            'inserted_q75': all_obj_q75,
            'inserted_iqr': all_obj_iqr,
            'inserted_cv': all_obj_cv
        }
 
    return statistics


def create_fused_pointcloud_with_intensity(pc_remain, pc_bg_intensity, pcd_obj_numpy, instance_mask, obj_names=None, bg_keep_mask=None, bg_remain_count=None, intensity_config=None):
    """Fuse background and object points while assigning per-point intensity values."""
    # Determine counts for background and object points
    if bg_remain_count is not None:
        bg_count = bg_remain_count
    else:
        bg_count = max(pc_remain.shape[0] - pcd_obj_numpy.shape[0], 0)

    # Derive intensity values for background points
    if len(pc_bg_intensity) == bg_count:
        bg_intensity = pc_bg_intensity
    elif bg_keep_mask is not None:
        if len(bg_keep_mask) != len(pc_bg_intensity):
            if len(bg_keep_mask) > len(pc_bg_intensity):
                bg_keep_mask = bg_keep_mask[:len(pc_bg_intensity)]
            else:
                padding = np.zeros(len(pc_bg_intensity) - len(bg_keep_mask), dtype=bool)
                bg_keep_mask = np.concatenate([bg_keep_mask, padding])

        bg_intensity = pc_bg_intensity[bg_keep_mask]

        if len(bg_intensity) != bg_count:
            if len(bg_intensity) > bg_count:
                bg_intensity = bg_intensity[:bg_count]
            else:
                remaining_needed = bg_count - len(bg_intensity)
                bg_intensity = np.concatenate([bg_intensity, pc_bg_intensity[:remaining_needed]])
    else:
        if bg_count > len(pc_bg_intensity):
            bg_count = len(pc_bg_intensity)
        bg_intensity = pc_bg_intensity[:bg_count]

    # Separate background and object point sets
    pc_bg_remain = pc_remain[:bg_count]
    pc_obj_remain = pc_remain[bg_count:]
    
    # Derive intensities for inserted object points using the configuration
    if obj_names is not None and len(obj_names) > 0:
        # Resolve the mapping from model name to intensity configuration (either provided or loaded)
        model_config_mapping = get_reflection_intensities_by_objects(intensity_config_dict=intensity_config)
        
        # Build a lookup from object name to its point cloud slice
        object_point_clouds = {}
        
        # Ensure instance_mask length matches pc_obj_remain (after occlusion filtering)
        if len(instance_mask) != len(pc_obj_remain):
            pass
            instance_mask = np.zeros(len(pc_obj_remain), dtype=int)
        
        for i, obj_name in enumerate(obj_names):
            obj_point_indices = np.where(instance_mask == i)[0]
            if len(obj_point_indices) > 0:
                obj_points = pc_obj_remain[obj_point_indices]
                # Use the instance id to disambiguate repeated asset names (e.g., composite objects)
                # Format: "obj_name_instance{i}" to mirror create_fused_pointcloud_with_intensity
                unique_key = f"{obj_name}_instance{i}"
                object_point_clouds[unique_key] = obj_points
                pass
        
        # Generate intensities via the shared helper
        pc_obj_with_intensity = pc_numpy_2_pcr_with_instance_intensity(pc_obj_remain, instance_mask, obj_names, model_config_mapping, object_point_clouds)
        obj_intensity = pc_obj_with_intensity[:, 3]  # Extract the intensity column
    else:
        # Fall back to default intensity assignment
        pc_obj_with_intensity = pc_numpy_2_pcr_with_instance_intensity(pc_obj_remain, instance_mask)
        obj_intensity = pc_obj_with_intensity[:, 3]  # Extract the intensity column
    
    # Concatenate background and object intensities
    fused_intensity = np.concatenate([bg_intensity, obj_intensity])
    
    if len(obj_intensity) > 0:
        _ = np.min(obj_intensity), np.max(obj_intensity)
    
    # Optionally compute statistics for reporting
    if obj_names is not None and len(obj_names) > 0:
        compare_intensity_statistics(
            pc_bg_intensity=bg_intensity,
            obj_names=obj_names,
            instance_mask=instance_mask,
            obj_intensity=obj_intensity,
            pc_bg_remain=pc_bg_remain,
            pc_obj_remain=pc_obj_remain
        )
        pass
    
    # Recombine spatial coordinates and intensities into an (N, 4) point cloud
    fused_pointcloud = np.concatenate([pc_remain, fused_intensity.reshape(-1, 1)], axis=1)
    
    return fused_pointcloud


def pc_numpy_2_pcr(mixed_pc_three_dims) -> np.ndarray:
    """Pad a 3D point cloud with a zero-intensity column."""
    # Ensure the input has three spatial dimensions
    assert mixed_pc_three_dims.shape[1] == 3
    # Number of points
    point_nums = mixed_pc_three_dims.shape[0]
    # TODO: allow callers to provide object intensities if available
    b = np.full((point_nums, 1), 0)
    mixed_pc = np.concatenate([mixed_pc_three_dims, b], axis=1)
    return mixed_pc


class PointCloud(object):
    # @Test status: Completed
    def __init__(self, point_cloud: np.ndarray = None, reflection: np.ndarray = None):
        self.point_cloud = point_cloud
        self.reflection = reflection
        self.kd_tree = None
    
    # @Test status: Completed
    def set_point_cloud(self, point_cloud: np.ndarray):
        self.point_cloud = point_cloud

    # @Test status: Completed
    def set_reflection(self, reflection: np.ndarray):
        self.reflection = reflection

    # @Test status: Completed
    def to_numpy_with_reflection(self):
        reflection_column = self.reflection.reshape(-1, 1)
        return np.concatenate((self.point_cloud, reflection_column), axis=1)

    def to_numpy(self):
        return self.point_cloud

    def get_reflection(self):
        return self.reflection

    # @Test status: Completed
    def save(self, path: str):
        np.save(path, self.to_numpy_with_reflection())

    # @Test status: Completed
    def save_xyz(self, path: str):
        np.save(path, self.point_cloud)

    # simple and fast visualization
    # @Test status: Completed
    def visualize(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        o3d.visualization.draw_geometries([pcd])

    # @Test status: Completed
    def visualize_with_reflection(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        reflection_values = self.reflection
        normalized_values = (reflection_values - np.min(reflection_values)) / (
                np.max(reflection_values) - np.min(reflection_values))
        colors = cm.viridis(normalized_values)[:, :-1]  # Apply the viridis colormap and drop the alpha channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    def get_kd_tree(self, refresh=False):
        from scipy.spatial import KDTree
        if self.kd_tree is None or refresh:
            self.kd_tree = KDTree(self.point_cloud[:, :2])
        return self.kd_tree

    def get_k_nearest_points(self, point, k, refresh=False):
        tree = self.get_kd_tree(refresh)
        distances, indices = tree.query(point, k=k)
        return distances, indices