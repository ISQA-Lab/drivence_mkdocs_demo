import os
import re
from drivence.utils.path_utils import get_project_root_dir


def get_obj_path_for_member(member, group_name):
    """Return the OBJ path for a configured asset member."""
    project_root = get_project_root_dir()
    obj_name = member["name"]
    mesh_dir = os.path.join(project_root, "drivence", "meshes", obj_name)
    obj_path = os.path.join(mesh_dir, "model_normalized.obj")
    return obj_path


def load_mesh_from_vehicle(vehicle, shapenet_loader):
    """Load a mesh for the given vehicle using available assets and fallbacks."""
    import os
    import glob
    import re
    from drivence.utils import mesh_utils
    
    # Prefer an explicitly specified mesh path on the vehicle
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
            scale_ratio = shapenet_loader._get_scale_ratio_with_name_and_frame(mesh_obj, None, vehicle.obj_name, frame_number)
        else:
            # Otherwise fall back to the default scaling
            scale_ratio = shapenet_loader._get_scale_ratio_with_name(mesh_obj, None, vehicle.obj_name)

        # Apply standard alignment/normalization
        car_mesh = shapenet_loader._normalize_mesh(mesh_obj, scale_ratio=scale_ratio, location=None, rotation=None)

    else:
        # Fallback to the generic model provided by shapenet_loader
        car_mesh = shapenet_loader.load_mesh_by_name(vehicle.obj_name)

    return car_mesh
