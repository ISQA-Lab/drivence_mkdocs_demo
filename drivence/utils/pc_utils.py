from drivence.utils.format_convert import numpy_to_torch
import numpy as np
import open3d as o3d
from matplotlib import cm
from lidar_core.intensity_formula.Formula import lambert_beckmann_model, eq_7, data


def load_pc_xyzr(path: str):
    return load_pc(path, 4)


def load_pc(path: str, demension=4, format=None):
    """
    加载点云文件（支持 .npy 和 .bin 格式），自动识别格式或按指定格式解析。

    核心逻辑：
    1. 若未指定格式，通过文件后缀自动识别（.npy → "npy"，.bin → "bin"）；
    2. 调用 _load_pc_detail 执行具体加载逻辑，返回指定维度的点云数组。

    Args:
        path (str): 点云文件路径（绝对路径或相对路径）；
        demension (int, 可选): 点云每个点的维度（如4维：X/Y/Z/强度），默认4；
        format (Optional[str], 可选): 指定文件格式（"npy" 或 "bin"），默认 None（自动识别）。

    Returns:
        np.ndarray: 加载的点云数组，形状为 (N, demension)，N为点云数量，数据类型为 float32。
    """
    if format is None:
        if path.split(".")[-1] == "npy":
            format = "npy"
        elif path.split(".")[-1] == "bin":
            format = "bin"
        else:
            raise ValueError(f"Unsupported file format: {path.split('.')[-1]}")
    return _load_pc_detail(path, demension, format)


def _load_pc_detail(path: str, demension, format):
    """
    点云加载核心函数，根据指定格式和维度加载点云（内部使用，不建议外部直接调用）。

    Args:
        path (str): 点云文件路径；
        demension (int): 点云维度；
        format (str): 文件格式（"npy" 或 "bin"）。

    Returns:
        np.ndarray: 点云数组，形状为 (N, demension)，数据类型为 float32。
    """
    if format == "npy":
        example = np.load(path).astype(np.float32)
    else:
        example = np.fromfile(path, dtype=np.float32).reshape(-1, demension)
    return example


def save_point_cloud(file_path, point_cloud):
    point_cloud.astype(np.float32).tofile(file_path)  # Final point cloud must be stored as float32 to guarantee it can be reloaded


def rotate_points_along_z(points: np.ndarray, angle: np.ndarray, return_type: str = "numpy") -> np.ndarray:
    """
    绕 Z 轴旋转批量点云（支持 numpy 数组和 torch 张量输入）。

    核心逻辑：
    1. 将输入转换为 torch 张量（统一计算逻辑）；
    2. 构建 Z 轴旋转矩阵（基于旋转角的余弦和正弦）；
    3. 对点云的 XYZ 坐标应用旋转，保留其他维度（如强度）；
    4. 按指定格式返回结果（numpy 或 torch 张量）。

    Args:
        points (np.ndarray): 输入点云数组，形状为 (B, N, D)：
            - B：批量大小；
            - N：每个点云的点数；
            - D：点云维度（≥3，前3维为 XYZ 坐标）；
        angle (np.ndarray): 旋转角数组，形状为 (B,)，单位为弧度；
        return_type (str, 可选): 返回格式，默认 "numpy"：
            - "numpy"：返回 numpy 数组；
            - "torch"：返回 torch 张量。

    Returns:
        Union[np.ndarray, torch.Tensor]: 旋转后的点云，形状与输入一致。
    """
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
    """
    从配置中加载每个模型的反射强度参数（物理参数），支持传入字典或从 YAML 文件读取。

    核心逻辑：
    1. 优先使用传入的配置字典，否则尝试加载 legacy YAML 文件（未实现）；
    2. 解析配置中的默认参数和各模型参数，构建模型名称到参数的映射；
    3. 处理配置缺失或解析失败的情况，返回空字典。

    Args:
        intensity_config_dict (Optional[dict], 可选): 强度配置字典，默认 None。

    Returns:
        Dict[str, dict]: 模型名称到强度参数的映射，每个参数字典包含：
            - p_lambda: 波长相关参数；
            - kd: 漫反射系数；
            - m: 表面粗糙度参数；
            - theta_T: 阈值角度（度）；
            解析失败时返回空字典。
    """
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
    """
    通过拟合查询点周围的切平面，估计表面法向量和入射角（激光雷达到点的射线与法向量的夹角）。

    核心逻辑：
    1. 若点云数量不足，返回默认法向量；
    2. 查找查询点的 k 个最近邻，计算邻域的协方差矩阵；
    3. 协方差矩阵的最小特征值对应的特征向量即为法向量；
    4. 调整法向量方向指向激光雷达（原点），计算锐角入射角。

    Args:
        point_xyz (np.ndarray): 查询点的 XYZ 坐标，形状为 (3,)；
        object_points (np.ndarray): 物体点云数组，形状为 (M, 3)，M为物体点数；
        k_neighbors (int, 可选): 用于拟合平面的最近邻数量，默认10。

    Returns:
        Tuple[np.ndarray, float]:
            - 表面法向量，形状为 (3,)，已归一化并指向激光雷达；
            - 入射角（度），范围 0–90 度。
    """
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
    """
    基于物理辐射测量模型（Lambert-Beckmann 混合模型）计算点的反射强度，参考公式 Eq.15。

    核心逻辑：
    1. 加载激光雷达系统硬件参数（如发射功率、接收孔径等）；
    2. 计算点到激光雷达的距离和表面入射角（通过切平面拟合估计法向量）；
    3. 结合模型物理参数（漫反射系数、表面粗糙度等）和系统参数，计算原始强度；
    4. 归一化强度到 0–1 范围，适配激光雷达传感器的动态范围。

    Args:
        model_name (str): 物体模型名称（用于调试或日志，不影响计算）；
        point_xyz (np.ndarray): 目标点的 3D 坐标（LiDAR 坐标系下），形状为 (3,)；
        model_config (dict): 模型反射强度物理参数字典，需包含：
            - p_lambda: 波长相关衰减系数（默认 0.7）；
            - kd: 漫反射系数（默认 0.95）；
            - m: 表面粗糙度参数（默认 0.15）；
            - theta_T: 角度阈值（度，默认 10）；
        object_points (Optional[np.ndarray], 可选): 物体的完整点云（用于估计表面法向量），
            形状为 (M, 3)，M 为物体点数，默认 None。

    Returns:
        float: 归一化后的反射强度（范围 0.0–1.0）。
    """
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
    """
    基于实例分割掩码，为 3D 点云添加逐点反射强度（支持物理模型计算或默认值）。

    核心逻辑：
    1. 校验输入点云维度（必须为 3 维 XYZ 坐标）；
    2. 遍历每个点，根据实例掩码获取对应的模型名称；
    3. 基于模型的物理参数，通过物理模型计算强度；
    4. 无配置时使用默认强度，最终返回 (N, 4) 维点云（XYZ+强度）。

    Args:
        mixed_pc_three_dims (np.ndarray): 3D 点云数组（仅 XYZ 坐标），形状为 (N, 3)，N 为点数；
        instance_mask (np.ndarray): 实例分割掩码，形状为 (N,)，每个元素为实例索引（对应 obj_names）；
        obj_names (Optional[List[str]], 可选): 实例名称列表，索引与 instance_mask 对应，默认 None；
        model_intensity_mapping (Optional[Dict[str, dict]], 可选): 模型名称到物理参数字典的映射，
            由 get_reflection_intensities_by_objects() 返回，默认 None；
        object_point_clouds (Optional[Dict[str, np.ndarray]], 可选): 实例点云字典，键为实例唯一标识，
            值为该实例的点云数组（用于估计法向量），默认 None。

    Returns:
        np.ndarray: 带强度的点云数组，形状为 (N, 4)，前 3 维为 XYZ 坐标，第 4 维为反射强度（0.0–1.0）。
    """
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
    """
    计算插入物体的反射强度描述性统计信息（按物体类型和整体汇总）。

    核心逻辑：
    1. 按物体名称分组，聚合同类物体的所有点强度；
    2. 计算每组的统计指标（均值、标准差、分位数等）；
    3. 计算所有插入物体的整体统计指标；
    4. 标记无点云数据的物体，返回统计结果字典。

    Args:
        pc_bg_intensity (np.ndarray): 背景点的强度数组，形状为 (B,)，B 为背景点数；
        obj_names (List[str]): 插入物体的名称列表；
        instance_mask (np.ndarray): 实例分割掩码，形状为 (O,)，O 为插入物体点数；
        obj_intensity (np.ndarray): 插入物体的强度数组，形状为 (O,)；
        pc_bg_remain (np.ndarray): 背景点云（XYZ 坐标），形状为 (B, 3)；
        pc_obj_remain (np.ndarray): 插入物体点云（XYZ 坐标），形状为 (O, 3)。

    Returns:
        Dict[str, dict]: 强度统计结果字典，包含：
            - 每个物体的统计指标（count、mean、std、min、max、median、q25、q75、iqr、cv）；
            - 整体统计指标（inserted_count、inserted_mean、inserted_std 等）；
            - 无点云物体的警告信息。
    """
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
    """
    融合背景点云和插入物体点云，为所有点分配反射强度（背景保留原始强度，物体基于物理模型计算）。

    核心逻辑：
    1. 确定背景点和物体点的数量，提取背景原始强度；
    2. 处理背景强度的长度匹配（裁剪或填充）；
    3. 为插入物体点计算强度（基于实例配置）；
    4. 融合背景和物体的强度，返回 (N, 4) 维融合点云。

    Args:
        pc_remain (np.ndarray): 待融合的点云（背景+物体，仅 XYZ 坐标），形状为 (N, 3)；
        pc_bg_intensity (np.ndarray): 原始背景点的强度数组，形状为 (B_orig,)；
        pcd_obj_numpy (np.ndarray): 插入物体的点云（仅 XYZ 坐标），形状为 (O, 3)；
        instance_mask (np.ndarray): 物体实例分割掩码，形状为 (O,)；
        obj_names (Optional[List[str]], 可选): 物体名称列表，默认 None；
        bg_keep_mask (Optional[np.ndarray], 可选): 背景点保留掩码（布尔数组），形状为 (B_orig,)，默认 None；
        bg_remain_count (Optional[int], 可选): 保留的背景点数量，默认 None；
        intensity_config (Optional[dict], 可选): 强度配置字典（传递给 get_reflection_intensities_by_objects），默认 None。

    Returns:
        np.ndarray: 融合后的点云数组，形状为 (N, 4)，前 3 维为 XYZ 坐标，第 4 维为反射强度。
    """
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
    """
    为 3D 点云（XYZ 坐标）添加零值强度列，转换为 4D 点云（XYZ+强度）。

    Args:
        mixed_pc_three_dims (np.ndarray): 3D 点云数组（仅 XYZ 坐标），形状为 (N, 3)。

    Returns:
        np.ndarray: 4D 点云数组，形状为 (N, 4)，第 4 维为 0（默认强度）。
    """
    # Ensure the input has three spatial dimensions
    assert mixed_pc_three_dims.shape[1] == 3
    # Number of points
    point_nums = mixed_pc_three_dims.shape[0]
    # TODO: allow callers to provide object intensities if available
    b = np.full((point_nums, 1), 0)
    mixed_pc = np.concatenate([mixed_pc_three_dims, b], axis=1)
    return mixed_pc


class PointCloud(object):
    """
    点云数据管理类，支持点云存储、反射强度管理、可视化、KD-Tree近邻查询和文件读写。

    核心功能：
    1. 点云（XYZ）和反射强度（Reflection）的存储与读写；
    2. 基础可视化（纯点云、反射强度着色可视化）；
    3. 2D KD-Tree构建与k近邻查询（基于X-Y坐标）；
    4. 点云文件保存（含强度的完整格式、纯XYZ格式）。
    """
    # @Test status: Completed
    def __init__(self, point_cloud: np.ndarray = None, reflection: np.ndarray = None):
        """
        初始化点云对象。

        Args:
            point_cloud (Optional[np.ndarray], 可选): 3D点云数组，形状为 (N, 3)，N为点数，
                每行为 [X, Y, Z] 坐标，默认 None；
            reflection (Optional[np.ndarray], 可选): 反射强度数组，形状为 (N,) 或 (N, 1)，
                与点云点数一一对应，默认 None。
        """
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
        """
        合并点云和反射强度，返回 (N, 4) 格式的点云数组（X-Y-Z-反射强度）。

        Returns:
            np.ndarray: 合并后的点云数组，形状为 (N, 4)。
        """
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
        """
        基于反射强度着色可视化点云（使用 viridis 色彩映射）。

        核心逻辑：
        1. 归一化反射强度到 0–1 范围；
        2. 使用 viridis 色彩映射为每个点分配颜色；
        3. 去除颜色的 alpha 通道，适配 Open3D 格式。
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        reflection_values = self.reflection
        normalized_values = (reflection_values - np.min(reflection_values)) / (
                np.max(reflection_values) - np.min(reflection_values))
        colors = cm.viridis(normalized_values)[:, :-1]  # Apply the viridis colormap and drop the alpha channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    def get_kd_tree(self, refresh=False):
        """
        构建或获取基于 X-Y 坐标的 2D KD-Tree（用于快速近邻查询）。

        Args:
            refresh (bool, 可选): 是否强制重建 KD-Tree，默认 False（复用缓存）。

        Returns:
            KDTree: 基于 X-Y 坐标的 KD-Tree 对象。
        """
        from scipy.spatial import KDTree
        if self.kd_tree is None or refresh:
            self.kd_tree = KDTree(self.point_cloud[:, :2])
        return self.kd_tree

    def get_k_nearest_points(self, point, k, refresh=False):
        """
        基于 X-Y 坐标查询 k 个最近邻点。

        Args:
            point (Union[np.ndarray, Tuple[float, float]]): 查询点的 X-Y 坐标，格式为 (2,) 数组或二元组；
            k (int): 需查询的近邻点数（k ≥ 1）；
            refresh (bool, 可选): 是否强制重建 KD-Tree，默认 False。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - 距离数组：形状为 (k,)，存储查询点到每个近邻点的距离；
                - 索引数组：形状为 (k,)，存储近邻点在原始点云中的索引。
        """
        tree = self.get_kd_tree(refresh)
        distances, indices = tree.query(point, k=k)
        return distances, indices