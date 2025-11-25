import os

from drivence.dataset.common.dataframe_factory import DataFrameFactory

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import math
from entry.containers import DataGenContainer
from drivence.entity.scene_info import SceneInfo

from drivence.utils.box_utils import box_o3d_to_corners3d, covert_boxo3d_to_aligned_boxo3d


def lidar_pipeline(ray_casting_lidar, label_lidar_gen, lidar_scene_info, shapenet_loader, dataset_config,
                   intensity_config=None):
    """
    LiDAR 点云数据生成流水线，实现背景点云与目标点云的融合、遮挡处理、语义标注及标准化输出。

    该流水线整合了「网格加载→点云仿真→遮挡计算→语义标签融合→强度配置→数据保存」全流程，
    支持从数据集读取背景点云与标定信息，通过 ray casting 仿真目标（如车辆）点云，自动处理目标与背景的
    遮挡关系，融合语义/实例标签，最终生成符合数据集格式要求的标准化 LiDAR 输出，适配跨模态融合、
    模型训练或仿真验证场景。

    核心流程：
    1. 加载场景网格与边界框：从场景信息中获取目标（如车辆）的 3D 网格及边界框数据；
    2. 初始化数据集与输出容器：根据配置创建输入数据集读取器和输出数据容器；
    3. 读取基础数据：获取背景点云、原始语义标签、标定信息及点云强度；
    4. 目标点云仿真：通过 ray casting 生成目标（如车辆）的点云及实例掩码；
    5. 语义标签生成：为仿真目标分配语义标签，适配数据集标签体系；
    6. 遮挡处理：计算目标与背景的遮挡关系，保留可见点云及对应标签；
    7. 标签融合：合并背景与目标的语义标签，处理实例掩码映射；
    8. 点云融合与强度配置：融合可见背景点云与目标点云，应用强度配置；
    9. 标准化输出：将融合后的点云、标签、标定信息存入输出容器并保存。

    Args:
        ray_casting_lidar (object): Ray casting  LiDAR 仿真器对象，需提供以下方法/属性：
            - simulate_all(car_list, return_rays, return_instance_mask): 仿真目标点云，返回点云数组与实例掩码；
            - handle_occlusion(...): 处理目标与背景的遮挡关系，返回可见点云及标签；
            - yaw_rotation_counterclockwise (float): LiDAR 逆时针偏航旋转角度（用于网格对齐）。
        label_lidar_gen (object): LiDAR 标签生成器对象，需提供：
            - generate_labels(labels, bounding_box_dicts, calib_info, lidar_scene_info):
              基于原始标签、边界框和标定信息生成融合标签。
        lidar_scene_info (object): LiDAR 场景信息对象，需包含以下属性/字段：
            - bg_index (int): 背景索引（用于数据集读取）；
            - semantic_labels (list/array, 可选): 目标语义标签列表（索引与实例掩码对应）；
            - obj_names (list, 可选): 目标名称列表（用于强度配置）；
            其他场景相关元信息（如场景坐标系、目标位置等）。
        shapenet_loader (object): ShapeNet 网格加载器对象，用于加载场景中目标的 3D 网格（适配 get_mesh_and_bounding_box 函数）。
        dataset_config (object): 数据集配置对象，需提供：
            - get_config(key): 获取指定配置项，需支持读取 "source_dataset"（源数据集配置）和 "generated_dataset"（生成数据集配置）；
            源数据集配置需包含 "dataset_type"（数据集类型）和 "dataset_output_type"（输出数据类型）。
        intensity_config (dict, 可选): 点云强度配置字典（默认 None），用于自定义目标/背景点云的强度计算规则，
            具体键值需适配 create_fused_pointcloud_with_intensity 函数。

    Returns:
        None: 函数无显式返回值，最终结果通过 dataset_output.save()。
    """
    lidar = ray_casting_lidar
    lidar_label_generator = label_lidar_gen

    bounding_box_dicts, car_list = get_mesh_and_bounding_box(lidar_scene_info, shapenet_loader,
                                                             lidar.yaw_rotation_counterclockwise)

    dataset_type = dataset_config.get_config("source_dataset").get("dataset_type", "")
    dataset_output_type = dataset_config.get_config("source_dataset").get("dataset_output_type", "")

    dataset = DataFrameFactory.get_data_frame(dataset_type, lidar_scene_info.bg_index,
                                              dataset_config.get_config("source_dataset"))
    dataset_output = DataFrameFactory.get_output_data_frame(dataset_type, dataset_output_type,
                                                            lidar_scene_info.bg_index,
                                                            dataset_config.get_config("generated_dataset"))

    labels = dataset.get_label()
    calib_info = dataset.get_calibration()
    background_pointcloud = dataset.get_point_cloud()
    pc_bg = background_pointcloud.point_cloud
    pc_bg_intensity = background_pointcloud.reflection

    from lidar_core.occlusion.semantic_label_handler import create_semantic_label_handler
    semantic_handler = create_semantic_label_handler(dataset_type)
    original_semantic_labels = semantic_handler.get_original_semantic_labels(dataset)

    pcd_obj_numpy, instance_mask = lidar.simulate_all(car_list, return_rays=False, return_instance_mask=True)

    semantic_labels = getattr(lidar_scene_info, "semantic_labels", None)
    obj_names = getattr(lidar_scene_info, "obj_names", None)

    vehicle_semantic_labels = semantic_handler.generate_vehicle_semantic_labels(
        pcd_obj_numpy, instance_mask=instance_mask, semantic_labels=semantic_labels
    )

    pc_remain, semantic_labels_remain, bg_keep_mask, bg_remain_count, intensity_bg_remain, instance_mask_remain = ray_casting_lidar.handle_occlusion(
        pc_bg=pc_bg, pc_obj=pcd_obj_numpy, mesh_obj_list=car_list,
        labels_bg=original_semantic_labels, labels_obj=vehicle_semantic_labels,
        intensity_bg=pc_bg_intensity,
        instance_mask_obj=instance_mask,
        return_bg_keep_mask=True
    )

    mix_labels = None
    if labels is not None and calib_info is not None:
        mix_labels = lidar_label_generator.generate_labels(labels, bounding_box_dicts, calib_info, lidar_scene_info)

    if semantic_labels_remain is not None:
        fused_semantic_labels = semantic_labels_remain
    else:
        if original_semantic_labels is not None and bg_keep_mask is not None:
            bg_labels = original_semantic_labels[bg_keep_mask]
        else:
            bg_labels = np.zeros(bg_remain_count, dtype=np.uint32)

        obj_remain_count = len(pc_remain) - bg_remain_count
        if obj_remain_count > 0:
            if instance_mask_remain is not None and semantic_labels is not None:
                obj_labels = np.empty(obj_remain_count, dtype=np.uint32)
                for i, inst in enumerate(instance_mask_remain):
                    if 0 <= inst < len(semantic_labels):
                        obj_labels[i] = semantic_labels[inst]
                    else:
                        obj_labels[i] = semantic_handler.vehicle_label_id
            else:
                obj_labels = np.full(obj_remain_count, semantic_handler.vehicle_label_id, dtype=np.uint32)
        else:
            obj_labels = np.empty(0, dtype=np.uint32)

        fused_semantic_labels = np.concatenate([bg_labels, obj_labels], axis=0)

    from drivence.utils.pc_utils import create_fused_pointcloud_with_intensity
    fused_pointcloud = create_fused_pointcloud_with_intensity(
        pc_remain, intensity_bg_remain, pcd_obj_numpy, instance_mask_remain, obj_names,
        bg_keep_mask=None,
        bg_remain_count=bg_remain_count,
        intensity_config=intensity_config
    )

    dataset_output.set(
        label=mix_labels,
        calib=calib_info,
        point_cloud=fused_pointcloud
    )
    semantic_handler.set_semantic_labels_to_output(dataset_output, fused_semantic_labels)
    dataset_output.save()


def get_mesh_and_bounding_box(lidar_scene_info, shapenet_loader, yaw_rotation_counterclockwise):
    """
    加载场景中所有车辆的 3D 网格模型（mesh）并计算其对齐后的边界框，为 LiDAR 仿真提供目标几何信息。

    遍历场景中的每辆车辆，通过网格加载器获取适配的 3D 网格，根据车辆的位置、旋转信息完成网格的空间对齐，
    并计算网格的最小有向边界框（OBB），转换为轴对齐边界框（AABB）后提取角点坐标，最终返回所有车辆的
    边界框字典列表与对齐后的网格列表，用于后续 LiDAR 点云仿真、遮挡计算及标签生成。

    核心流程：
    1. 遍历场景车辆列表：逐个处理 lidar_scene_info 中的每辆车辆；
    2. 加载车辆网格：调用 load_mesh_from_vehicle 加载适配的 3D 网格；
    3. 网格空间对齐：根据车辆的位置（平移）和旋转角度（绕 Z 轴）调整网格姿态，适配 LiDAR 坐标系；
    4. 边界框计算与转换：先获取网格的最小有向边界框，再转换为轴对齐边界框并提取角点；
    5. 结果收集：汇总所有车辆的边界框信息（名称+角点）和对齐后的网格，返回给调用方。

    Args:
        lidar_scene_info (object): LiDAR 场景信息对象，需包含：
            - vehicles (list): 车辆对象列表，每个车辆对象需具备：
                - location (list/tuple/np.ndarray): 车辆位置坐标，格式为 (x, y, z)；
                - rotation (float): 车辆绕 Z 轴的旋转角度（单位：度）；
                - obj_name (str): 车辆对应的网格名称（用于边界框信息标记）。
        shapenet_loader (object): ShapeNet 网格加载器对象，用于加载车辆的 3D 网格（适配 load_mesh_from_vehicle 函数）。
        yaw_rotation_counterclockwise (bool): 偏航旋转方向标识：
            - True: 车辆旋转角度按逆时针方向计算；
            - False: 车辆旋转角度按顺时针方向计算（将输入旋转角度转换为 2π - 弧度值）。

    Returns:
        tuple: 包含两个元素的元组：
            1. bounding_box_dicts (list[dict]): 边界框信息字典列表，每个字典包含：
                - "obj_name" (str): 车辆对应的网格名称；
                - "corners" (np.ndarray): 轴对齐边界框的 8 个角点坐标，形状为 (8, 3)，
                  每个角点格式为 (x, y, z)，坐标系统与 LiDAR 一致。
            2. car_list (list[object]): 对齐后的车辆 3D 网格模型列表，每个网格对象需支持：
                - translate(location): 平移网格到指定位置；
                - get_rotation_matrix_from_xyz(angles): 根据 (x, y, z) 角度生成旋转矩阵；
                - rotate(rotation_matrix): 应用旋转矩阵调整网格姿态；
                - get_minimal_oriented_bounding_box(): 获取网格的最小有向边界框（OBB）。
    """
    bounding_box_dicts = []
    car_list = []

    from drivence.assets.assets_utils import load_mesh_from_vehicle

    for vehicle in lidar_scene_info.vehicles:
        car_mesh = load_mesh_from_vehicle(vehicle, shapenet_loader)

        car_mesh.translate(vehicle.location)
        rz_radians = math.radians(vehicle.rotation)
        if not yaw_rotation_counterclockwise:
            rz_radians = 2 * np.pi - rz_radians
        RZ = car_mesh.get_rotation_matrix_from_xyz((0, 0, rz_radians))
        car_mesh.rotate(RZ)
        car_list.append(car_mesh)

        obj_inserted_box3d = car_mesh.get_minimal_oriented_bounding_box()
        obj_box3d_adjusted = covert_boxo3d_to_aligned_boxo3d(obj_inserted_box3d)
        obj_box3d_corners = box_o3d_to_corners3d(obj_box3d_adjusted)
        bounding_box_dicts.append({"obj_name": vehicle.obj_name, "corners": np.array(obj_box3d_corners)})

    return bounding_box_dicts, car_list


def get_lidar_scene_info():
    ...


def run(task_name, lidar_scene_info):
    """
    LiDAR 数据生成任务入口函数，负责初始化依赖组件、解析场景信息并触发数据生成流水线。

    作为顶层调度函数，该函数接收任务名称和场景信息，完成场景信息标准化解析、数据生成所需组件（虚拟 LiDAR、标签生成器、网格加载器等）的初始化，
    读取强度配置（可选），最终调用 lidar_pipeline 执行完整的数据生成流程（点云仿真、遮挡处理、标签融合、输出保存），适配批量数据生成或单场景测试场景。

    核心流程：
    1. 任务信息打印：输出当前执行的任务名称，便于日志追踪；
    2. 场景信息解析：将输入的场景信息（字符串/字典格式）标准化为 SceneInfo 对象，统一后续处理接口；
    3. 组件初始化：通过 DataGenContainer 容器初始化数据集配置、虚拟 LiDAR、标签生成器、ShapeNet 网格加载器等核心组件；
    4. 强度配置加载：尝试从容器中读取强度配置（失败时静默忽略，使用默认配置）；
    5. 触发流水线：调用 lidar_pipeline 传入所有初始化组件和配置，执行 LiDAR 数据生成全流程。

    Args:
        task_name (str): 任务名称，用于日志输出和任务标识（如“train_data_gen_001”“test_scene_01”），不直接参与业务逻辑。
        lidar_scene_info (str/dict/SceneInfo): 场景信息输入，支持三种格式：
            - str: 场景配置文件路径（如 JSON/YAML 文件），用于从文件加载场景信息；
            - dict: 场景配置字典，直接包含车辆列表、背景索引等核心信息；
            - SceneInfo: 已实例化的场景信息对象，直接复用无需解析。
            最终会统一转换为 SceneInfo 对象供后续流水线使用。

    Returns:
        None: 函数无显式返回值，数据生成结果由 lidar_pipeline 中的 dataset_output.save() 保存到指定路径（由数据集配置指定）。
    """
    print("task_name:", task_name)
    if isinstance(lidar_scene_info, str) or isinstance(lidar_scene_info, dict):
        lidar_scene_info = SceneInfo(lidar_scene_info)

    container = DataGenContainer()
    dataset_config = container.dataset_config()

    intensity_config = None
    try:
        if container.intensity_config is not None:
            intensity_config_obj = container.intensity_config()
            intensity_config = intensity_config_obj.get_config()
    except (AttributeError, KeyError, TypeError):
        pass

    ray_casting_lidar = container.virtual_lidar()
    label_lidar_gen = container.lidar_label_generator()
    shapenet_loader = container.shapenet_loader()

    lidar_pipeline(ray_casting_lidar, label_lidar_gen, lidar_scene_info, shapenet_loader, dataset_config,
                   intensity_config)


if __name__ == '__main__':
    pass
