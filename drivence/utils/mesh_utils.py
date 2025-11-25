import math
import numpy as np
import open3d as o3d

from drivence.utils.format_convert import pc_numpy_2_pcd

def get_mesh_scale_ratio(mesh_obj,target_lwh,adaptive_length=False):
    """
    计算网格模型（mesh）的缩放比例，使缩放后的模型尺寸适配目标长宽高（LWH），采用最小缩放比确保完全适配。

    核心逻辑：
    1. 提取网格模型的原始尺寸（长度、宽度、高度）；
    2. 分别计算目标尺寸与原始尺寸在长、宽、高三个维度的缩放比；
    3. 选取三个维度中的最小缩放比（确保缩放后模型不会超出目标尺寸，完全适配）；
    4. 校验输入和计算结果的合法性，避免无效缩放。

    Args:
        mesh_obj (Union[o3d.geometry.TriangleMesh, Any]): 输入的网格模型对象，
            需支持 `get_mesh_info` 函数的解析（返回包含 width/length/height 的字典）；
        target_lwh (Tuple[float, float, float]): 目标尺寸，格式为 (target_length, target_width, target_height)，
            三个维度均需为正数（单位需与网格原始尺寸一致）；
        adaptive_length (bool, 可选): `get_mesh_info` 函数的参数，控制是否自适应调整长度计算逻辑，
            具体行为由 `get_mesh_info` 定义，默认 False。

    Returns:
        float: 网格模型的缩放比例（正数），将模型的长、宽、高分别乘以该比例后，
            各维度尺寸均≤目标尺寸，且至少一个维度等于目标尺寸。
    """
    info = get_mesh_info(mesh_obj,adaptive_length)
    width = info["width"]
    length = info["length"]
    height = info["height"]
    target_length,target_width,target_height = target_lwh
    if target_length <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target_lwh: {target_lwh}. Each dimension must be > 0.")
    if width <= 0 or length <= 0 or height <= 0:
        raise ValueError(f"Invalid mesh dimensions: width={width}, length={length}, height={height}.")
    scale_ratio_width = target_width / width
    scale_ratio_length = target_length / length
    scale_ratio_height = target_height / height
    scale_ratio = min(scale_ratio_width,scale_ratio_length,scale_ratio_height)
    if scale_ratio <= 0 or not np.isfinite(scale_ratio):
        raise ValueError(f"Computed invalid scale_ratio: {scale_ratio} from target_lwh={target_lwh} and mesh dims (LWH)=({length},{width},{height})")
    return scale_ratio


def _get_translation_vector(mesh_obj, target_bottom_center=None):
    """
    计算网格模型（mesh）的平移向量，使模型的底部中心对齐到目标位置（默认Z轴贴合地面）。

    核心逻辑：
    1. 计算网格模型的轴对齐边界（最大/最小边界）；
    2. 基于边界计算模型的底部中心（X-Y平面中心，Z轴取最小边界）；
    3. 若指定目标底部中心，计算当前底部中心到目标位置的平移向量；
    4. 保留Z轴平移分量（确保模型底部贴合地面或目标Z坐标）。

    Args:
        mesh_obj (o3d.geometry.TriangleMesh): Open3D网格模型对象（需已初始化，包含几何边界信息）；
        target_bottom_center (Optional[np.ndarray], 可选): 目标底部中心的3D坐标，形状为 (3,)，
            格式为 [x_target, y_target, z_target]，默认 None（此时平移向量Z轴分量为0，仅X-Y轴居中）。

    Returns:
        np.ndarray: 平移向量，形状为 (3,)，格式为 [tx, ty, tz]，
            网格模型沿该向量平移后，底部中心将对齐到目标位置（或默认位置）。
    """
    max_bound = np.array(mesh_obj.get_max_bound())
    min_bound = np.array(mesh_obj.get_min_bound())
    bottom_center = np.array([(max_bound[0] + min_bound[0]) / 2,  (max_bound[1] + min_bound[1]) / 2,min_bound[2]])
    translation_vector = target_bottom_center - bottom_center
    translation_vector[2] = translation_vector[2] # wheel in ground
    return translation_vector
  


def load_mesh(mesh_path) -> o3d.t.geometry.TriangleMesh:
    """
    从指定路径加载3D网格模型（TriangleMesh），支持Open3D兼容的格式（如.obj、.ply等）。

    核心逻辑：
    1. 使用Open3D的read_triangle_mesh读取网格文件；
    2. 检查网格是否为空，为空则打印警告并返回None；
    3. 返回有效网格对象（注：原函数注释返回类型为o3d.t.geometry.TriangleMesh，实际返回经典版TriangleMesh，已修正注释）。

    Args:
        mesh_path (str): 网格文件路径（绝对路径或相对路径），支持格式：.obj、.ply、.stl等Open3D兼容格式。

    Returns:
        Optional[o3d.geometry.TriangleMesh]: 加载成功的网格对象；若文件路径错误或网格为空，返回None。
    """
    mesh_obj = o3d.io.read_triangle_mesh(mesh_path)
    if mesh_obj.is_empty():
        print("warning: can't load mesh from path {},chek path is correct".format(mesh_path))
        return None
    return mesh_obj

def get_mesh_info(mesh_obj,adaptive_length=False):
    """
    提取网格模型的关键尺寸信息（长宽高、对角线、中心），支持自适应调整长宽顺序。

    核心逻辑：
    1. 计算网格的轴对齐边界（最小/最大XYZ坐标）；
    2. 基于边界计算宽度（Y轴跨度）、长度（X轴跨度）、高度（Z轴跨度）；
    3. 计算XY平面对角线长度和几何中心；
    4. 可选：自适应调整长宽顺序（确保长度≥宽度），或仅打印警告。

    Args:
        mesh_obj (o3d.geometry.TriangleMesh): Open3D网格模型对象（需已初始化，包含边界信息）；
        adaptive_length (bool, 可选): 是否自适应调整长宽顺序，默认False：
            - True：若长度<宽度，交换两者，确保长度≥宽度；
            - False：若长度<宽度，仅打印警告，不调整。

    Returns:
        Dict[str, Any]: 网格尺寸信息字典，包含：
            - "width": 宽度（Y轴跨度，单位与网格坐标一致）；
            - "length": 长度（X轴跨度，单位与网格坐标一致）；
            - "height": 高度（Z轴跨度，单位与网格坐标一致）；
            - "diagonal": XY平面对角线长度（单位与网格坐标一致）；
            - "center": 几何中心坐标（[x, y, z]，单位与网格坐标一致）。
    """
    min_xyz = mesh_obj.get_min_bound()
    max_xyz = mesh_obj.get_max_bound()
    x_min, x_max = min_xyz[0], max_xyz[0]
    y_min, y_max = min_xyz[1], max_xyz[1]
    z_min, z_max = min_xyz[2], max_xyz[2]
    width = (y_max - y_min)
    length = (x_max - x_min)
    height = (z_max - z_min)
    diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    if adaptive_length:
        if length < width:
            a = length
            length = width
            width = a
    else:
        if length < width:
            print("warning: mesh is not a regular car, length < width")      
    return {"width":width,"length":length,"height":height,"diagonal":diagonal,"center":center}


def get_hull_vertices(mesh_obj):
    """
    计算网格模型的凸包，并返回凸包的顶点坐标数组。

    核心逻辑：
    1. 对输入网格计算凸包（最紧凑的凸多边形包裹网格）；
    2. 提取凸包的顶点坐标，转换为numpy数组返回。

    Args:
        mesh_obj (o3d.geometry.TriangleMesh): Open3D网格模型对象（需包含顶点和面信息）。

    Returns:
        np.ndarray: 凸包顶点坐标数组，形状为 (M, 3)，
            M为凸包顶点数量，每行对应 [x, y, z] 坐标（与网格坐标系一致）。
    """
    hull, _ = mesh_obj.compute_convex_hull()
    obj_hull_vertices = np.asarray(hull.vertices)
    return obj_hull_vertices



def get_pyramid_mesh(centerCamPoint, hull_vertices) -> o3d.geometry.TriangleMesh:
    """
    生成金字塔形网格（以相机中心为顶点，凸包顶点为底面）。

    核心逻辑：
    1. 合并相机中心和凸包顶点，形成完整的点集；
    2. 对合并点集计算凸包，得到金字塔形网格（底面为凸包，顶点为相机中心）。

    Args:
        centerCamPoint (np.ndarray): 相机中心3D坐标，形状为 (3,)，格式为 [x, y, z]；
        hull_vertices (np.ndarray): 凸包顶点坐标数组，形状为 (M, 3)，M为凸包顶点数量。

    Returns:
        o3d.geometry.TriangleMesh: 金字塔形网格对象（凸包网格，底面贴合输入凸包，顶点指向相机）。
    """
    combinedVertices = np.vstack((hull_vertices, centerCamPoint))  
    center2obj_pcd_shadow = pc_numpy_2_pcd(combinedVertices)
    center2obj_shadow_mesh, _ = center2obj_pcd_shadow.compute_convex_hull()
    return center2obj_shadow_mesh


def get_frustum_mesh(centerCamPoint, hull_vertices) -> o3d.geometry.TriangleMesh:
    """
    生成视锥体（Frustum）网格（以相机中心为起点，向凸包顶点反向延伸形成的无限锥台的有限部分）。

    核心逻辑：
    1. 对每个凸包顶点，计算从相机中心指向该顶点的射线，并沿射线反向延伸100单位得到新顶点；
    2. 对延伸后的顶点集计算凸包，得到视锥体的"远截面"；
    3. 合并原始凸包顶点和延伸后凸包顶点，计算凸包得到完整视锥体网格。

    Args:
        centerCamPoint (np.ndarray): 相机中心3D坐标，形状为 (3,)，格式为 [x, y, z]；
        hull_vertices (np.ndarray): 凸包顶点坐标数组，形状为 (M, 3)，M为凸包顶点数量（构成视锥体近截面）。

    Returns:
        o3d.geometry.TriangleMesh: 视锥体网格对象（闭合凸包，包含近截面、远截面和侧面）。
    """
    cast_hull_points = np.asarray([])
    for point1 in hull_vertices:

        ba = centerCamPoint - point1
        baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
        ba2 = ba / baLen
        pt2 = centerCamPoint + ((-100) * ba2)

        if (np.size(cast_hull_points)):
            cast_hull_points = np.vstack((cast_hull_points, [pt2]))
        else:
            cast_hull_points = np.array([pt2])
    pcd_cast_hull = pc_numpy_2_pcd(cast_hull_points)
    
    hull2, _ = pcd_cast_hull.compute_convex_hull()
    cast_hull_vertices = np.asarray(hull2.vertices)
    
    combinedVertices = np.vstack((hull_vertices, cast_hull_vertices))
    pcd_shadow = pc_numpy_2_pcd(combinedVertices)
    shadow_mesh, _ = pcd_shadow.compute_convex_hull()
    return shadow_mesh