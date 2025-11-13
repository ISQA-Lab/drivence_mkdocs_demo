from entry.containers import DataGenContainer
from drivence.dataset.common.dataframe_factory import DataFrameFactory
"""SemanticKITTI pc-only data generation entrypoints"""
from drivence.entity.scene_info import SceneInfo
from drivence.module.data_gen import main
from drivence.module.logic_scene.logic_scene_generator import create_logic_scene_generator
from drivence.module.pose_gen.pose_generation import generate_group_pose_from_road
from drivence.utils.common_utils import UUIDManager
from drivence.utils.path_utils import get_project_root_dir
from drivence.assets.assets_utils import get_obj_path_for_member
import argparse
import yaml
import os
import json


SURFACE_PREFIX_MAP = [
    ("scooter", "sidewalk"),
    ("person", "sidewalk"),
    ("pedestrian", "sidewalk"),
    ("personPullingLuggage", "sidewalk"),
    ("personPushingBicycle", "sidewalk"),
    ("car", "road"),
    ("truck", "road"),
    ("bus", "road"),
    ("barrier", "sidewalk"),
    ("tractor", "road"),
    ("man", "sidewalk"),
    ("police", "sidewalk"),
    ("worker", "sidewalk"),
    ("senior", "sidewalk"),
    ("kid", "sidewalk"),
    ("trunk", "terrain"),
    ("leaves", "terrain"),
    ("shrubbery", "terrain"),
    ("pole", "sidewalk"),
    ("sign", "sidewalk"),
    ("streetlight", "sidewalk"),
    ("telephonepole", "sidewalk"),
    ("dog", "sidewalk"),
    ("boar", "sidewalk"),
    ("cat", "sidewalk"),
    ("luggage", "sidewalk"),
    ("stroller", "sidewalk"),
    ("wheelchair", "sidewalk")
]

def category_of(name: str) -> str:
    """
    Map an object name to its preferred surface type.

    Args:
        name: Object identifier such as "Car_1" or "person_1".

    Returns:
        Target surface type: "road", "sidewalk", or "terrain".
    """
    lower = name.lower()
    for prefix, surface in SURFACE_PREFIX_MAP:
        if lower.startswith(prefix):
            return surface
    return "road"  # default surface

def generate_data_with_point_cloud(bg_index: int, obj_name: str, count: int = 1):
    """
    Insert one or more copies of an object into a SemanticKITTI frame and regenerate point-cloud labels.

    Args:
        bg_index: Background frame index.
        obj_name: Object identifier (for example "Car_1", "Bus_935").
        count: Number of instances to insert (default: 1).
    """
    bg_index = int(bg_index)
    count = int(count)
    if count < 1:
        raise ValueError("count must be a positive integer")

    #print(f"[DEBUG] Start inserting {count} object(s) of {obj_name} into background frame {bg_index}")
    
    # Acquire road split utilities from the container
    container = DataGenContainer()
    road_split = container.road_split()
    shapenet_loader = container.shapenet_loader()
    dataset_config = container.dataset_config()

    dataset_type = dataset_config.get_config("source_dataset").get("dataset_type", "")
    dataset_output_type = dataset_config.get_config("source_dataset").get("dataset_output_type", "")
    source_dataset_cfg = dataset_config.get_config("source_dataset")

    # Load object configuration
    obj_insert_config_rel_path = source_dataset_cfg.get("obj_insert_config_path", "configs/obj_insert_config.yml")
    if not os.path.isabs(obj_insert_config_rel_path):
        project_root = get_project_root_dir()
        obj_insert_config_path = os.path.join(project_root, obj_insert_config_rel_path)
    else:
        obj_insert_config_path = obj_insert_config_rel_path

    # Load YAML/JSON configuration
    with open(obj_insert_config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    config_file = yaml_config.get("config_file", "obj_insert_config.json")
    if not os.path.isabs(config_file):
        config_dir = os.path.dirname(obj_insert_config_path)
        json_config_path = os.path.join(config_dir, config_file)
    else:
        json_config_path = config_file
    
    with open(json_config_path, "r") as f:
        json_config = json.load(f)

    groups_cfg = yaml_config.get("groups", [])
    groups_detail = json_config.get("groups", {})

    # Look up the member that matches obj_name
    member = None
    group_name_found = None
    for group_name, group_detail in groups_detail.items():
        for m in group_detail.get("members", []):
            if m["name"] == obj_name:
                member = m
                group_name_found = group_name
                break
        if member:
            break
        
    if member is None:
        print(f"[WARN] No explicit configuration found for {obj_name}; falling back to defaults")
        semantic_label = 252
        relative_displacement = [0, 0, 0]
        relative_rotation = 0.0
        obj_mesh_path = None  # Use default path
    else:
        semantic_label = member.get("semantic_label", 252)
        relative_displacement = member.get("relative_displacement", [0, 0, 0])
        relative_rotation = member.get("relative_rotation", 0.0)
        obj_mesh_path = get_obj_path_for_member(member, group_name_found)
        #print(f"[DEBUG] Configuration found for {obj_name}, semantic_label={semantic_label}")

    scale_ratio = shapenet_loader.get_scale_ratio(obj_name)
    #print(f"[DEBUG] {obj_name} uses default scale ratio: {scale_ratio}")
        
    # Load mesh resources
    if member is not None and obj_mesh_path and os.path.exists(obj_mesh_path):
        # Handle custom OBJ assets generated under mesh_gen
        if "mesh_gen" in obj_mesh_path or "frame" in obj_mesh_path:
            #print(f"[DEBUG] Loading custom OBJ file: {obj_mesh_path}")
            from drivence.utils import mesh_utils
            mesh = mesh_utils.load_mesh(obj_mesh_path)
            if mesh is None:
                raise RuntimeError(f"Failed to load OBJ file: {obj_mesh_path}")
            mesh = shapenet_loader._align_mesh_in_lidar(mesh)
            mesh = shapenet_loader._normalize_mesh(mesh, scale_ratio=scale_ratio, location=None, rotation=None)
            #print(f"[DEBUG] Custom OBJ loaded and normalized: {obj_mesh_path}")
        else:
            #print(f"[DEBUG] Loading mesh via shapenet_loader (path exists but needs normalization)")
            mesh = shapenet_loader.load_mesh_by_name(obj_name)
    else:
        #print(f"[DEBUG] Loading default mesh from shapenet_loader configuration")
        mesh = shapenet_loader.load_mesh_by_name(obj_name)
        
    # Multi-object configuration
    mesh_list = [mesh] * count
    obj_names = [obj_name] * count
    semantic_labels = [semantic_label] * count
    relative_displacement_list = [list(relative_displacement) for _ in range(count)]
    relative_rotation_list = [relative_rotation for _ in range(count)]
    scale_ratio_list = [scale_ratio for _ in range(count)]
    group_members_indices = [[i] for i in range(count)]
    if member is not None and obj_mesh_path and os.path.exists(obj_mesh_path) and ("mesh_gen" in obj_mesh_path or "frame" in obj_mesh_path):
        obj_paths = [obj_mesh_path for _ in range(count)]
    else:
        default_path = shapenet_loader.get_mesh_path(obj_name)
        obj_paths = [default_path for _ in range(count)]

    dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, source_dataset_cfg)
    dataset_output = DataFrameFactory.get_output_data_frame(dataset_type, dataset_output_type, bg_index,
                                                            dataset_config.get_config("generated_dataset"))

    surface_type = category_of(obj_name)
    if surface_type == "sidewalk":
        surface_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "sidewalk_split_label")
    elif surface_type == "terrain":
        surface_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "terrain_split_label")
    else:  # "road"
        surface_label_dir = dataset.road_split_label_dir
    
    surface_pc, non_surface_pc = road_split.split_pcd_road(bg_index, dataset.get_point_cloud_path(),
                                                          surface_label_dir,
                                                          surface_label_dir,
                                                          insert_location=surface_type,
                                                          semantic_label_dir=dataset.semantic_label_dir)
    
    if surface_pc is None or len(surface_pc) <= 10:
        print(f"[WARN] {surface_type} point cloud unavailable or too sparse; falling back to road")
        surface_pc, non_surface_pc = road_split.split_pcd_road(bg_index, dataset.get_point_cloud_path(),
                                                                 dataset.road_split_label_dir,
                                                                 dataset.road_split_label_dir,
                                                                 insert_location="road",
                                                                 semantic_label_dir=dataset.semantic_label_dir)
        if surface_pc is None or len(surface_pc) <= 10:
            raise RuntimeError(f"Failed to obtain valid road point cloud")
    
    calib_info = None
    surface_pc_valid = surface_pc

    obj_lidar_positions, obj_rz_degrees = generate_group_pose_from_road(
        surface_pc_valid, non_surface_pc, calib_info, dataset, mesh_list,
        group_members_indices, relative_displacement_list, relative_rotation_list
    )

    success_indices = [i for i, pos in enumerate(obj_lidar_positions) if pos is not None]
    if not success_indices:
        raise RuntimeError(f"Failed to insert {obj_name} into the scene")
    if len(success_indices) < count:
        print(f"[WARN] Successfully inserted {len(success_indices)}/{count} object(s); some placements failed")

    obj_lidar_positions = [obj_lidar_positions[i] for i in success_indices]
    obj_rz_degrees = [obj_rz_degrees[i] for i in success_indices]
    obj_names = [obj_names[i] for i in success_indices]
    semantic_labels = [semantic_labels[i] for i in success_indices]
    scale_ratio_list = [scale_ratio_list[i] for i in success_indices]
    obj_paths = [obj_paths[i] for i in success_indices]

    generator = create_logic_scene_generator(dataset_type, dataset)
    lidar_scene_info = generator.convert_pose_to_scene_infos(
        obj_lidar_positions, obj_rz_degrees,
        scale_ratio_list,
        [1] * len(obj_names), obj_names, "objects",
        bg_index,
        dataset_output.get_lidar_scene_info_path()
    )
    
    setattr(lidar_scene_info, "semantic_labels", semantic_labels)
    setattr(lidar_scene_info, "obj_names", obj_names)
    #print(f"[DEBUG] scene_info.semantic_labels set to {semantic_labels}")
    #print(f"[DEBUG] scene_info.obj_names set to {obj_names}")
    
    for i, vehicle in enumerate(lidar_scene_info.vehicles):
        if i < len(obj_paths) and obj_paths[i]:
            if os.path.exists(obj_paths[i]) and ("mesh_gen" in obj_paths[i] or "frame" in obj_paths[i]):
                vehicle.obj_path = obj_paths[i]
                #print(f"[DEBUG] Vehicle {obj_names[i]} obj_path set to {obj_paths[i]}")
            else:
                vehicle.obj_path = shapenet_loader.get_mesh_path(obj_names[i])
                #print(f"[DEBUG] Vehicle {obj_names[i]} obj_path set to {vehicle.obj_path}")
        else:
            vehicle.obj_path = shapenet_loader.get_mesh_path(obj_names[i])
            #print(f"[DEBUG] Vehicle {obj_names[i]} obj_path set to {vehicle.obj_path}")

    main.run(UUIDManager().get_uuid(), lidar_scene_info)


def generate_data_with_logic_scene(logic_scene_path: str):
    """
    Generate data from a pre-defined logic-scene JSON.

    Args:
        logic_scene_path: Path to the logic-scene file.
    """
    if not os.path.exists(logic_scene_path):
        raise FileNotFoundError(f"Logic-scene file not found: {logic_scene_path}")

    with open(logic_scene_path, "r", encoding='utf-8') as f:
        scene_data = json.load(f)

    print("==============================================================")
    print("Generating data from logic-scene file")
    print("==============================================================")
    print(f"Scene file: {logic_scene_path}")
    print(f"Background frame: {scene_data.get('bg_index', 'N/A')}")
    print(f"Objects: {scene_data.get('nums_vehicles', 0)}")

    container = DataGenContainer()
    dataset_config = container.dataset_config()
    source_dataset_cfg = dataset_config.get_config("source_dataset")

    obj_insert_config_rel_path = source_dataset_cfg.get("obj_insert_config_path",
                                                      "configs/obj_insert_config.yml")
    if not os.path.isabs(obj_insert_config_rel_path):
        project_root = get_project_root_dir()
        obj_insert_config_path = os.path.join(project_root, obj_insert_config_rel_path)
    else:
        obj_insert_config_path = obj_insert_config_rel_path

    groups_detail = {}
    groups_cfg = []
    if os.path.exists(obj_insert_config_path):
        with open(obj_insert_config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        config_file = yaml_config.get("config_file", "obj_insert_config.json")
        if not os.path.isabs(config_file):
            config_dir = os.path.dirname(obj_insert_config_path)
            json_config_path = os.path.join(config_dir, config_file)
        else:
            json_config_path = config_file

        if os.path.exists(json_config_path):
            with open(json_config_path, "r") as f:
                json_config = json.load(f)
            groups_detail = json_config.get("groups", {})
            groups_cfg = yaml_config.get("groups", [])

    scene_info = SceneInfo(scene_data)
    vehicle_semantic_labels = []
    vehicle_obj_names = []

    for vehicle in scene_info.vehicles:
        obj_name = vehicle.obj_name
        vehicle_obj_names.append(obj_name)

        semantic_label = None
        obj_path = None
        group_name_found = None

        for group_name, group_detail in groups_detail.items():
            for member in group_detail.get("members", []):
                if member["name"] == obj_name:
                    semantic_label = member.get("semantic_label", 98)
                    group_name_found = group_name
                    obj_path = get_obj_path_for_member(member, group_name)
                    print(f"✓ {obj_name}: semantic_label={semantic_label}, obj_path={obj_path}")
                    break
            if semantic_label is not None:
                break
        
        if semantic_label is None:
            semantic_label = 98
            print(f"[WARN] {obj_name}: configuration missing, using default semantic_label=98")
            shapenet_loader = container.shapenet_loader()
            try:
                obj_path = shapenet_loader.get_mesh_path(obj_name)
            except:
                obj_path = None

        vehicle.semantic_label = semantic_label
        vehicle_semantic_labels.append(semantic_label)
        if obj_path:
            vehicle.obj_path = obj_path

    setattr(scene_info, "semantic_labels", vehicle_semantic_labels)
    setattr(scene_info, "obj_names", vehicle_obj_names)
    print(f"\n✓ semantic_labels: {vehicle_semantic_labels}")
    print(f"✓ obj_names: {vehicle_obj_names}")
    print(f"✓ rotation preserved from JSON")
    print("==============================================================")
    print("")

    task_name = UUIDManager().get_uuid()
    main.run(task_name, scene_info)


def generate_data_with_group(bg_index: int):
    """
    Insert multiple objects according to group definitions.

    Args:
        bg_index: Background frame index.
    """
    #print(f"[DEBUG] Start group insertion, background frame: {bg_index}")

    container = DataGenContainer()
    road_split = container.road_split()
    shapenet_loader = container.shapenet_loader()
    dataset_config = container.dataset_config()

    dataset_type = dataset_config.get_config("source_dataset").get("dataset_type", "")
    dataset_output_type = dataset_config.get_config("source_dataset").get("dataset_output_type", "")
    source_dataset_cfg = dataset_config.get_config("source_dataset")

    obj_insert_config_rel_path = source_dataset_cfg.get("obj_insert_config_path", "configs/obj_insert_config.yml")
    if not os.path.isabs(obj_insert_config_rel_path):
        project_root = get_project_root_dir()
        obj_insert_config_path = os.path.join(project_root, obj_insert_config_rel_path)
    else:
        obj_insert_config_path = obj_insert_config_rel_path

    # Load YAML and JSON configuration files
    with open(obj_insert_config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    config_file = yaml_config.get("config_file", "obj_insert_config.json")
    if not os.path.isabs(config_file):
        config_dir = os.path.dirname(obj_insert_config_path)
        json_config_path = os.path.join(config_dir, config_file)
    else:
        json_config_path = config_file
    
    with open(json_config_path, "r") as f:
        json_config = json.load(f)

    groups_cfg = yaml_config.get("groups", [])
    groups_detail = json_config.get("groups", {})
    
    if not groups_cfg:
        raise ValueError("No active groups defined in obj_insert_config.yml")

    all_members_info = []
    group_members_indices = []
    current_index = 0
    
    for group_cfg in groups_cfg:
        group_name = group_cfg.get("group_name")
        if not group_name:
            continue
        
        if group_name not in groups_detail:
            print(f"[WARN] Group '{group_name}' not found in JSON; skipping")
            continue
        
        group_detail = groups_detail[group_name]
        members = group_detail.get("members", [])
        
        if not members:
            print(f"[WARN] Group '{group_name}' has no members; skipping")
            continue
        
        group_indices = []
        
        for member in members:
            obj_name = member.get("name")
            if not obj_name:
                continue
            
            member_info = {
                "obj_name": obj_name,
                "semantic_label": member.get("semantic_label", 252),
                "relative_displacement": member.get("relative_displacement", [0, 0, 0]),
                "relative_rotation": member.get("relative_rotation", 0.0),
                "obj_path": get_obj_path_for_member(member, group_name),
                "member": member,
                "group_name": group_name,
            }
            
            all_members_info.append(member_info)
            group_indices.append(current_index)
            current_index += 1
        
        if group_indices:
            group_members_indices.append(group_indices)
            #print(f"[DEBUG] Group '{group_name}' includes {len(group_indices)} members")

    if not all_members_info:
        raise ValueError("No valid members found in configuration")

    #print(f"[DEBUG] Preparing to insert {len(all_members_info)} objects across {len(group_members_indices)} groups")

    mesh_list = []
    obj_names = []
    semantic_labels = []
    relative_displacement_list = []
    relative_rotation_list = []
    scale_ratio_list = []
    obj_paths = []
    
    for member_info in all_members_info:
        obj_name = member_info["obj_name"]
        obj_names.append(obj_name)
        semantic_labels.append(member_info["semantic_label"])
        relative_displacement_list.append(member_info["relative_displacement"])
        relative_rotation_list.append(member_info["relative_rotation"])
        obj_paths.append(member_info["obj_path"])
        
        member = member_info["member"]
        scale_ratio = shapenet_loader.get_scale_ratio(obj_name)
        scale_ratio_list.append(scale_ratio)
        
        obj_mesh_path = member_info["obj_path"]
        if member_info["member"] is not None and obj_mesh_path and os.path.exists(obj_mesh_path):
            if "mesh_gen" in obj_mesh_path or "frame" in obj_mesh_path:
                #print(f"[DEBUG] Loading custom OBJ file: {obj_mesh_path}")
                from drivence.utils import mesh_utils
                mesh = mesh_utils.load_mesh(obj_mesh_path)
                if mesh is None:
                    raise RuntimeError(f"Failed to load OBJ file: {obj_mesh_path}")
                mesh = shapenet_loader._align_mesh_in_lidar(mesh)
                mesh = shapenet_loader._normalize_mesh(mesh, scale_ratio=scale_ratio, location=None, rotation=None)
            else:
                mesh = shapenet_loader.load_mesh_by_name(obj_name)
        else:
            mesh = shapenet_loader.load_mesh_by_name(obj_name)
        
        mesh_list.append(mesh)
    
    dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, source_dataset_cfg)
    dataset_output = DataFrameFactory.get_output_data_frame(dataset_type, dataset_output_type, bg_index,
                                                            dataset_config.get_config("generated_dataset"))

    road_pc, non_road_pc = road_split.split_pcd_road(bg_index, dataset.get_point_cloud_path(),
                                                     dataset.road_split_label_dir,
                                                     dataset.road_split_label_dir,
                                                     insert_location="road",
                                                     semantic_label_dir=dataset.semantic_label_dir)
    calib_info = None
    road_pc_valid = road_pc

    #print(f"[DEBUG] Generating poses for group members...")
    obj_lidar_positions, obj_rz_degrees = generate_group_pose_from_road(
        road_pc_valid, non_road_pc, calib_info, dataset, mesh_list,
        group_members_indices, relative_displacement_list, relative_rotation_list
    )

    success_indices = [i for i, p in enumerate(obj_lidar_positions) if p is not None]
    if not success_indices:
        raise RuntimeError("No objects were successfully inserted")

    #print(f"[DEBUG] Inserted {len(success_indices)}/{len(all_members_info)} objects successfully")

    obj_lidar_positions = [obj_lidar_positions[i] for i in success_indices]
    obj_rz_degrees = [obj_rz_degrees[i] for i in success_indices]
    obj_names = [obj_names[i] for i in success_indices]
    semantic_labels = [semantic_labels[i] for i in success_indices]
    scale_ratio_list = [scale_ratio_list[i] for i in success_indices]
    obj_paths = [obj_paths[i] for i in success_indices]
    
    generator = create_logic_scene_generator(dataset_type, dataset)
    lidar_scene_info = generator.convert_pose_to_scene_infos(
        obj_lidar_positions, obj_rz_degrees,
        scale_ratio_list,
        [1] * len(success_indices), obj_names, "objects",
        bg_index,
        dataset_output.get_lidar_scene_info_path()
    )
    
    setattr(lidar_scene_info, "semantic_labels", semantic_labels)
    setattr(lidar_scene_info, "obj_names", obj_names)
    #print(f"[DEBUG] scene_info.semantic_labels set to {semantic_labels}")
    #print(f"[DEBUG] scene_info.obj_names set to {obj_names}")
    
    for i, vehicle in enumerate(lidar_scene_info.vehicles):
        if i < len(obj_paths) and obj_paths[i]:
            if os.path.exists(obj_paths[i]) and ("mesh_gen" in obj_paths[i] or "frame" in obj_paths[i]):
                vehicle.obj_path = obj_paths[i]
            else:
                vehicle.obj_path = shapenet_loader.get_mesh_path(obj_names[i])
        else:
            vehicle.obj_path = shapenet_loader.get_mesh_path(obj_names[i])
    
    main.run(UUIDManager().get_uuid(), lidar_scene_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SemanticKITTI pc-only data generation')
    subparsers = parser.add_subparsers(dest='command')

    p_pc = subparsers.add_parser('pc', help='Insert single object into a background frame')
    p_pc.add_argument('bg_index', type=int)
    p_pc.add_argument('obj_name')

    p_scene = subparsers.add_parser('scene', help='Generate data from a logic scene JSON')
    p_scene.add_argument('logic_scene_path')

    p_group = subparsers.add_parser('group', help='Insert a configured group into a background frame')
    p_group.add_argument('bg_index', type=int)

    args = parser.parse_args()

    if args.command == 'pc':
        generate_data_with_point_cloud(args.bg_index, args.obj_name)
    elif args.command == 'scene':
        generate_data_with_logic_scene(args.logic_scene_path)
    elif args.command == 'group':
        generate_data_with_group(args.bg_index)
    else:
        print("No command provided. Running a quick pc-only demo: bg_index=0, obj_name=Car_1")
        generate_data_with_point_cloud(1, "Car_1")