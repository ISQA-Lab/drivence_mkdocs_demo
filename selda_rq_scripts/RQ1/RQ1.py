import importlib
import json
import os
import random
import re
import sys

from drivence.utils.path_utils import get_project_root_dir

PROJECT_ROOT = get_project_root_dir()

os.environ.setdefault("DRIVENCE_ROOT_DIR", PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entry.containers import DataGenContainer
from drivence.dataset.common.dataframe_factory import DataFrameFactory
from drivence.module.data_gen import main
from drivence.module.logic_scene.logic_scene_generator import create_logic_scene_generator
from drivence.utils.common_utils import UUIDManager
from drivence.module.pose_gen.pose_generation import generate_group_pose_from_road
from drivence.assets.assets_utils import get_obj_path_for_member
import yaml

def get_category_prefix(group_name: str) -> str:

    prefix = re.sub(r'_\d+$', '', group_name)
    return prefix.lower()

def get_unique_categories(groups_detail: dict) -> list:

    categories = set()
    for group_name in groups_detail.keys():
        category = get_category_prefix(group_name)
        categories.add(category)
    return sorted(categories)

def generate_data_fixed_category(
    bg_index: int,
    category: str = "car",
    num_objects: int = 5,
):

    container = DataGenContainer()
    road_split = container.road_split()
    shapenet_loader = container.shapenet_loader()
    dataset_config = container.dataset_config()

    dataset_type = dataset_config.get_config("source_dataset").get("dataset_type", "")
    dataset_output_type = dataset_config.get_config("source_dataset").get("dataset_output_type", "")
    source_dataset_cfg = dataset_config.get_config("source_dataset")

    obj_insert_config_rel_path = source_dataset_cfg.get("obj_insert_config_path", "configs/obj_insert_config.yml")
    if not os.path.isabs(obj_insert_config_rel_path):
        obj_insert_config_path = os.path.join(PROJECT_ROOT, obj_insert_config_rel_path)
    else:
        obj_insert_config_path = obj_insert_config_rel_path

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

    groups_detail = json_config.get("groups", {})

    category_groups = []
    for group_name in groups_detail.keys():
        group_category = get_category_prefix(group_name)
        if group_category == category.lower():
            category_groups.append(group_name)

    if not category_groups:
        raise ValueError(f"No group found for category '{category}'")

    if num_objects > len(category_groups):
        print(f"Warning: num_objects ({num_objects}) > available groups ({len(category_groups)}), using all available groups")
        selected_group_names = category_groups
    else:
        selected_group_names = random.sample(category_groups, num_objects)

    obj_names = []
    relative_displacement_list = []
    relative_rotation_list = []
    semantic_label_list = []
    obj_mesh_paths = []
    group_members_indices = []

    for group_name in selected_group_names:
        group_detail = groups_detail.get(group_name, {})
        members = group_detail.get("members", [])
        member_indices = []
        for member in members:
            obj_names.append(member["name"])
            relative_displacement_list.append(member.get("relative_displacement", [0, 0, 0]))
            relative_rotation_list.append(member.get("relative_rotation", 0.0))
            semantic_label_list.append(member.get("semantic_label", 252))
            obj_mesh_paths.append(get_obj_path_for_member(member, group_name))
            member_indices.append(len(obj_names) - 1)
        group_members_indices.append(member_indices)

    mesh_list = []
    scale_ratio_arr = []
    objs_index_arr = []
    for i, obj_name in enumerate(obj_names):

        member = None
        for group_name, group_detail in groups_detail.items():
            for m in group_detail.get("members", []):
                if m["name"] == obj_name:
                    member = m
                    break
            if member:
                break
        if member and "fbx_frame" in member:
            frame_number = member["fbx_frame"]
            scale_ratio = shapenet_loader.get_scale_ratio_by_frame(obj_name, frame_number)
        else:
            scale_ratio = shapenet_loader.get_scale_ratio(obj_name)

        scale_ratio_arr.append(scale_ratio)

        mesh = shapenet_loader.load_mesh_by_name(obj_name)
        mesh_list.append(mesh)
        objs_index_arr.append(i + 1)

    dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, source_dataset_cfg)

    dataset_output = DataFrameFactory.get_output_data_frame(
        dataset_type, dataset_output_type, bg_index, dataset_config.get_config("generated_dataset")
    )

    calib_info = dataset.get_calibration()

    road_pc, non_road_pc = road_split.split_pcd_road(
        bg_index,
        dataset.get_point_cloud_path(),
        dataset.road_split_label_dir,
        dataset.road_split_label_dir,
    )
    if road_pc is None or len(road_pc) == 0:
        raise RuntimeError(f"Failed to obtain road points for bg_index={bg_index}. Please ensure road split labels exist.")

    if calib_info is not None:
        pts_img, pts_rect_depth = calib_info.lidar_to_img(road_pc)
        road_pc_valid = road_split.get_pc_road_in_img(pts_img, pts_rect_depth, road_pc)
    else:
        road_pc_valid = road_pc

    sidewalk_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "sidewalk_split_label")
    sidewalk_pc_valid = None
    sidewalk_non_road_pc = None
    if os.path.isdir(sidewalk_label_dir):
        sw_pc, sw_non = road_split.split_pcd_road(
            bg_index,
            dataset.get_point_cloud_path(),
            sidewalk_label_dir,
            sidewalk_label_dir,
            "sidewalk"
        )
        sidewalk_pc_valid = sw_pc
        sidewalk_non_road_pc = sw_non

    terrain_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "terrain_split_label")
    terrain_pc_valid = None
    terrain_non_road_pc = None
    if os.path.isdir(terrain_label_dir):
        tr_pc, tr_non = road_split.split_pcd_road(
            bg_index,
            dataset.get_point_cloud_path(),
            terrain_label_dir,
            terrain_label_dir,
            "terrain"
        )
        if calib_info is not None and tr_pc is not None:
            tr_img, tr_depth = calib_info.lidar_to_img(tr_pc)
            terrain_pc_valid = road_split.get_pc_road_in_img(tr_img, tr_depth, tr_pc)
        else:
            terrain_pc_valid = tr_pc
        terrain_non_road_pc = tr_non

    SURFACE_PREFIX_MAP = [
        ("bicycle", "sidewalk"),
        ("motorcycle", "sidewalk"),

        ("scooter", "sidewalk"),
        ("person", "sidewalk"),
        ("bicyclist", "sidewalk"),
        ("motorcyclist", "sidewalk"),
        ("pedestrian", "sidewalk"),
        ("personPullingLuggage", "sidewalk"),
        ("personPushingBicycle", "sidewalk"),
        ("car", "road"),
        ("truck", "road"),
        ("bus", "road"),
        ("barrier", "sidewalk"),
        ("motorcycle", "road"),
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
        lower = name.lower()
        for prefix, surface in SURFACE_PREFIX_MAP:
            if lower.startswith(prefix):
                return surface
        return "road"

    road_groups = []
    sidewalk_groups = []
    terrain_groups = []
    for member_indices in group_members_indices:
        if not member_indices:
            continue
        first_member_name = obj_names[member_indices[0]]
        surface_type = category_of(first_member_name)
        if surface_type == "sidewalk" and sidewalk_pc_valid is not None:
            sidewalk_groups.append(member_indices)
        elif surface_type == "terrain" and terrain_pc_valid is not None:
            terrain_groups.append(member_indices)
        else:
            road_groups.append(member_indices)

    full_obj_lidar_positions = [None for _ in range(len(mesh_list))]
    full_obj_rz_degrees = [None for _ in range(len(mesh_list))]

    def run_subset(sub_groups, use_pc_valid, use_non_road):
        if not sub_groups:
            return
        sub_positions, sub_rot = generate_group_pose_from_road(
            use_pc_valid, use_non_road, calib_info, dataset,
            mesh_list, sub_groups, relative_displacement_list, relative_rotation_list
        )
        for group in sub_groups:
            for idx in group:
                if sub_positions[idx] is not None:
                    full_obj_lidar_positions[idx] = sub_positions[idx]
                if sub_rot[idx] is not None:
                    full_obj_rz_degrees[idx] = sub_rot[idx]

    run_subset(road_groups, road_pc_valid, non_road_pc)
    if sidewalk_groups and (sidewalk_pc_valid is not None):
        use_non = sidewalk_non_road_pc if (sidewalk_non_road_pc is not None) else non_road_pc
        run_subset(sidewalk_groups, sidewalk_pc_valid, use_non)
    if terrain_groups and (terrain_pc_valid is not None):
        use_non = terrain_non_road_pc if (terrain_non_road_pc is not None) else non_road_pc
        run_subset(terrain_groups, terrain_pc_valid, use_non)

    obj_lidar_positions, obj_rz_degrees = full_obj_lidar_positions, full_obj_rz_degrees

    keep_indices_set = set()
    for member_indices in group_members_indices:
        if not member_indices:
            continue

        all_ok = all(
            (0 <= idx < len(obj_lidar_positions)) and (obj_lidar_positions[idx] is not None)
            for idx in member_indices
        )
        if all_ok:
            keep_indices_set.update(member_indices)

    success_indices = sorted(list(keep_indices_set))
    if not success_indices:
        raise RuntimeError("no object inserted in this frame")

    obj_lidar_positions = [obj_lidar_positions[i] for i in success_indices]
    obj_rz_degrees = [obj_rz_degrees[i] for i in success_indices]
    obj_names = [obj_names[i] for i in success_indices]
    obj_mesh_paths = [obj_mesh_paths[i] for i in success_indices]
    semantic_label_list = [semantic_label_list[i] for i in success_indices]
    scale_ratio_arr = [scale_ratio_arr[i] for i in success_indices]
    objs_index_arr = [i + 1 for i in range(len(success_indices))]

    generator = create_logic_scene_generator(dataset_type, dataset)
    lidar_scene_info = generator.convert_pose_to_scene_infos(
        obj_lidar_positions, obj_rz_degrees,
        scale_ratio_arr,
        objs_index_arr, obj_names, "objects",
        bg_index,
        dataset_output.get_lidar_scene_info_path()
    )

    setattr(lidar_scene_info, "semantic_labels", semantic_label_list)
    setattr(lidar_scene_info, "obj_names", obj_names)
    for i, vehicle in enumerate(lidar_scene_info.vehicles):
        if i < len(obj_mesh_paths):
            vehicle.obj_path = obj_mesh_paths[i]

    main.run(UUIDManager().get_uuid(), lidar_scene_info)

    try:
        return sum(1 for p in obj_lidar_positions if p is not None)
    except Exception:
        return 0

if __name__ == '__main__':

    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['fire', 'Fire']:
        try:
            fire_module = importlib.import_module("fire")
        except ImportError as exc:
            raise RuntimeError("Google Fire is not installed; please 'pip install fire' or omit the 'fire' argument.") from exc

        fire_module.Fire({
            'generate_data_fixed_category': generate_data_fixed_category,
        })
        sys.exit(0)

    print("=" * 60)
    print("RQ1: automated multi-round experiment")
    print("=" * 60)

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = PROJECT_ROOT

    config_path = os.path.join(project_root, "configs/obj_insert_config.json")
    with open(config_path, "r") as f:
        json_config = json.load(f)

    groups_detail = json_config.get("groups", {})
    all_categories = get_unique_categories(groups_detail)

    default_categories = [
        "car", "bus", "truck", "tractor", "scooter", "bicycle", "motorcycle",
        "ambulance", "adult", "senior", "police", "worker", "kid",
        "fence", "street_light", "traffic_sign", "trafficcone",
        "telephone_pole", "shrubbery", "tree", "dog", "cat", "boar",
        "luggage", "stroller", "wheelchair", "person-riding-bicycle",
        "person-riding-motor", "person-pushing-bicycle",
        "person-pushing-motorcycle", "person-pulling-luggage", "dog-walker"
    ]

    env_categories = os.environ.get("SELDA_RQ1_CATEGORIES")
    if env_categories:
        selected_categories = [c.strip().lower() for c in env_categories.split(",") if c.strip()]
    else:
        selected_categories = [c.lower() for c in default_categories]

    categories = [cat for cat in selected_categories if cat in all_categories]
    missing_categories = sorted(set(selected_categories) - set(categories))

    print(f"Total available categories: {len(all_categories)}")
    print(f"Selected categories: {len(categories)}")
    if missing_categories:
        print(f"Warning: the following categories are not defined and will be skipped: {missing_categories}")

    bg_file_name = os.environ.get("SELDA_RQ1_BGS_FILE", "bgs_rq1_0.txt")
    bg_file_path = os.path.join(tools_dir, bg_file_name)
    if not os.path.exists(bg_file_path):
        raise FileNotFoundError(f"Missing background index list: {bg_file_path}")

    bg_indices = []
    with open(bg_file_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                value = int(s)
            except ValueError:
                continue
            if 0 <= value < 4071:
                bg_indices.append(value)

    if not bg_indices:
        raise RuntimeError(f"Background index list {bg_file_path} is empty")

    required_bg_indices = [1441, 1442, 1443, 1444, 1445]
    if all(bg in bg_indices for bg in required_bg_indices):
        print(f"Required background frames present: {required_bg_indices}")
    else:
        missing = [bg for bg in required_bg_indices if bg not in bg_indices]
        print(f"Warning: missing required background frames {missing}")

    num_objects = int(os.environ.get("SELDA_RQ1_NUM_OBJECTS", "5"))

    print(f"Total background frames: {len(bg_indices)}")

    for category_idx, category in enumerate(categories, start=1):
        print(f"\nProcessing category {category_idx}/{len(categories)}: {category}")

        success_count = 0
        for frame_idx, bg_index in enumerate(bg_indices, start=1):
            try:
                generate_data_fixed_category(
                    bg_index=bg_index,
                    category=category,
                    num_objects=num_objects,
                )
                success_count += 1
            except Exception as e:
                print(f"Error at bg_index={bg_index}: {e}")
                continue

        print(
            f"Category '{category}' summary: success {success_count}/{len(bg_indices)}, "
            f"failed {len(bg_indices) - success_count}"
        )

    print("\n" + "=" * 60)
    print("RQ1 experiment finished")
    print("=" * 60)
