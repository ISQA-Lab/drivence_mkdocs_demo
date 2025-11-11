import math
import numpy as np
import open3d as o3d

from drivence.utils.format_convert import pc_numpy_2_pcd

def get_mesh_scale_ratio(mesh_obj,target_lwh,adaptive_length=False):
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
    """Calculate translation vector to align mesh bottom center with target position."""
    max_bound = np.array(mesh_obj.get_max_bound())
    min_bound = np.array(mesh_obj.get_min_bound())
    bottom_center = np.array([(max_bound[0] + min_bound[0]) / 2,  (max_bound[1] + min_bound[1]) / 2,min_bound[2]])
    translation_vector = target_bottom_center - bottom_center
    translation_vector[2] = translation_vector[2] # wheel in ground
    return translation_vector
  


def load_mesh(mesh_path) -> o3d.t.geometry.TriangleMesh:
    mesh_obj = o3d.io.read_triangle_mesh(mesh_path)
    if mesh_obj.is_empty():
        print("warning: can't load mesh from path {},chek path is correct".format(mesh_path))
        return None
    return mesh_obj

def get_mesh_info(mesh_obj,adaptive_length=False):
    """Get mesh dimensions. If adaptive_length is True, swap length and width if needed."""
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
    hull, _ = mesh_obj.compute_convex_hull()
    obj_hull_vertices = np.asarray(hull.vertices)
    return obj_hull_vertices



def get_pyramid_mesh(centerCamPoint, hull_vertices) -> o3d.geometry.TriangleMesh:
    combinedVertices = np.vstack((hull_vertices, centerCamPoint))  
    center2obj_pcd_shadow = pc_numpy_2_pcd(combinedVertices)
    center2obj_shadow_mesh, _ = center2obj_pcd_shadow.compute_convex_hull()
    return center2obj_shadow_mesh


def get_frustum_mesh(centerCamPoint, hull_vertices) -> o3d.geometry.TriangleMesh:
    """Generate frustum mesh from camera center and hull vertices."""
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