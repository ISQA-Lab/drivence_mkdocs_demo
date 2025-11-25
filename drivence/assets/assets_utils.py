import os
import re
from drivence.utils.path_utils import get_project_root_dir


def get_obj_path_for_member(member, group_name):
    """
    获取配置化资产成员（member）对应的 OBJ 模型文件路径。

    基于项目根目录、资产成员名称（member name），拼接出标准化 OBJ 模型文件的绝对路径，
    该文件默认存储在项目指定的 meshes 目录下，且文件名固定为 "model_normalized.obj"。
    适用于需要加载特定资产成员对应的 3D 模型文件的场景（如可视化、仿真、模型解析）。

    Args:
        member (dict): 资产成员配置字典，必须包含 "name" 键。
            - "name" (str): 资产成员名称，用于定位对应的模型目录（meshes/[name]）；
            若字典中无 "name" 键，会触发 KeyError 异常。
        group_name (str): 资产成员所属的组名称（预留参数，当前版本未直接参与路径拼接，
            用于后续扩展组级目录划分或权限校验）。

    Returns:
        str: OBJ 模型文件的绝对路径，格式为：
            [项目根目录]/drivence/meshes/[member.name]/model_normalized.obj
            路径已通过 os.path.join 适配不同操作系统（Windows/Linux/macOS）的路径分隔符。
    """
    project_root = get_project_root_dir()
    obj_name = member["name"]
    mesh_dir = os.path.join(project_root, "drivence", "meshes", obj_name)
    obj_path = os.path.join(mesh_dir, "model_normalized.obj")
    return obj_path


def load_mesh_from_vehicle(vehicle, shapenet_loader):
    """
    从车辆（vehicle）对象加载 3D 网格模型（mesh），支持多优先级加载策略与自动对齐归一化。

    采用「显式路径优先 → 预生成网格 fallback → ShapeNet 通用模型兜底」的三级加载逻辑，
    加载后自动完成网格与激光雷达（LiDAR）的空间对齐、尺度缩放及标准化处理，确保输出模型
    适配 Li-Fusion 等跨模态融合场景的坐标系统与尺度要求。

    核心流程：
    1. 优先使用 vehicle 对象中显式指定的 OBJ 路径（若存在且有效）；
    2. 若未指定显式路径，从项目工作目录的 mesh_gen 文件夹中搜索预生成的帧相关网格；
    3. 若前两级均未找到，通过 shapenet_loader 加载 ShapeNet 库中的通用模型；
    4. 加载后自动执行对齐（lidar 坐标系）、尺度计算、归一化等后处理步骤。

    Args:
        vehicle (object): 车辆对象，需包含以下属性/字段：
            - obj_path (str, 可选): 显式指定的 OBJ 网格文件路径，若存在且文件有效则优先使用；
            - obj_name (str, 必需): 车辆对应的网格名称（用于搜索预生成文件或 ShapeNet 通用模型）；
            若缺少 obj_name 属性，会导致后续加载逻辑失败。
        shapenet_loader (object): ShapeNet 网格加载器对象，需提供以下方法：
            - _align_mesh_in_lidar(mesh): 将网格对齐到激光雷达坐标系；
            - _get_scale_ratio_with_name_and_frame(mesh, location, obj_name, frame_number):
              根据网格名称和帧号计算尺度缩放比例；
            - _get_scale_ratio_with_name(mesh, location, obj_name):
              根据网格名称计算默认尺度缩放比例；
            - _normalize_mesh(mesh, scale_ratio, location, rotation): 对网格进行尺度归一化；
            - load_mesh_by_name(obj_name): 根据名称加载 ShapeNet 通用网格模型。

    Returns:
        object: 经过对齐、尺度缩放和归一化处理后的 3D 网格模型对象，适配 LiDAR 坐标系，
            可直接用于后续跨模态特征融合、可视化或仿真计算。
            网格模型的具体类型由 mesh_utils.load_mesh 和 shapenet_loader 的返回类型决定（如 trimesh.Trimesh）。
    """
    # Prefer an explicitly specified mesh path on the vehicle
    import os
    import glob
    import re
    from drivence.utils import mesh_utils
    mesh_gen_obj_path = None
    frame_number = None

    if hasattr(vehicle, 'obj_path') and vehicle.obj_path and os.path.exists(vehicle.obj_path):
        mesh_gen_obj_path = vehicle.obj_path

        # Extract frame number from the path if present
        frame_match = re.search(r'frame(\d+)', mesh_gen_obj_path)
        if frame_match:
            frame_number = int(frame_match.group(1))
    else:
        # Fallback: look for pre-generated meshes under mesh_gen
        project_root = get_project_root_dir()
        mesh_gen_dir = os.path.join(project_root, "drivence", "_workplace", "mesh_gen")

        # Filenames follow: {group_name}_{obj_name}_frame{frame_number}.obj
        pattern = os.path.join(mesh_gen_dir, f"*_{vehicle.obj_name}_frame*.obj")
        matches = glob.glob(pattern)

        if matches:
            # Use the first match
            mesh_gen_obj_path = matches[0]

            # Extract frame number from the filename
            frame_match = re.search(r'frame(\d+)', mesh_gen_obj_path)
            if frame_match:
                frame_number = int(frame_match.group(1))

    # Load the mesh asset
    if mesh_gen_obj_path and os.path.exists(mesh_gen_obj_path):
        mesh_obj = mesh_utils.load_mesh(mesh_gen_obj_path)
        mesh_obj = shapenet_loader._align_mesh_in_lidar(mesh_obj)

        # Choose the scaling strategy based on whether a frame override exists
        if frame_number is not None:
            # Reuse frame-specific dimensions when available
            scale_ratio = shapenet_loader._get_scale_ratio_with_name_and_frame(mesh_obj, None, vehicle.obj_name,
                                                                               frame_number)
        else:
            # Otherwise fall back to the default scaling
            scale_ratio = shapenet_loader._get_scale_ratio_with_name(mesh_obj, None, vehicle.obj_name)

        # Apply standard alignment/normalization
        car_mesh = shapenet_loader._normalize_mesh(mesh_obj, scale_ratio=scale_ratio, location=None, rotation=None)

    else:
        # Fallback to the generic model provided by shapenet_loader
        car_mesh = shapenet_loader.load_mesh_by_name(vehicle.obj_name)

    return car_mesh
