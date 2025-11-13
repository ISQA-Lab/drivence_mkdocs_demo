

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from drivence.utils.path_utils import get_project_root_dir

PROJECT_ROOT = get_project_root_dir()

os.environ.setdefault("DRIVENCE_ROOT_DIR", PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entry.containers import DataGenContainer
from drivence.assets.assets_utils import get_obj_path_for_member
from drivence.dataset.common.dataframe_factory import DataFrameFactory
from drivence.entity.scene_info import SceneInfo
from drivence.module.data_gen import main
from drivence.module.logic_scene.logic_scene_generator import create_logic_scene_generator
from drivence.module.pose_gen.pose_generation import generate_group_pose_from_road
from drivence.utils.common_utils import UUIDManager

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_BGS_FILE = TOOLS_DIR / "bgs_rq2_0.txt"
ERROR_FILE = TOOLS_DIR / "error_rq2.txt"
ORIGINAL_OUTPUT_FILE = TOOLS_DIR / "rq2_original_diversity.json"
GENERATED_OUTPUT_FILE = TOOLS_DIR / "rq2_generated_diversity.json"
MAX_BG_INDEX = 4070

SEMANTIC_KITTI_LABELS = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}

LEARNING_MAP = {
    0: 0,
    1: 0,
    10: 1,
    11: 2,
    13: 5,
    15: 3,
    16: 5,
    18: 4,
    20: 5,
    30: 6,
    31: 7,
    32: 8,
    40: 9,
    44: 10,
    48: 11,
    49: 12,
    50: 13,
    51: 14,
    52: 0,
    60: 9,
    70: 15,
    71: 16,
    72: 17,
    80: 18,
    81: 19,
    98: 20,
    99: 20,
    252: 1,
    253: 7,
    254: 6,
    255: 8,
    256: 5,
    257: 5,
    258: 4,
    259: 5,
}

LEARNING_LABELS = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
    20: "other-object"
}

LEARNING_IGNORE = {
    0: True,
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
    6: False,
    7: False,
    8: False,
    9: True,
    10: True,
    11: True,
    12: True,
    13: True,
    14: False,
    15: False,
    16: False,
    17: True,
    18: False,
    19: False,
    20: False
}

def map_labels_to_learning_labels(semantic_labels: np.ndarray) -> np.ndarray:

    if semantic_labels.dtype == np.uint32:
        semantic_labels = semantic_labels & 0xFFFF

    learning_labels = np.zeros_like(semantic_labels, dtype=np.uint8)
    for orig_label, learn_label in LEARNING_MAP.items():
        learning_labels[semantic_labels == orig_label] = learn_label

    return learning_labels

def compute_gini_diversity(semantic_labels: np.ndarray,
                          ignore_unlabeled: bool = True) -> float:

    if semantic_labels is None or len(semantic_labels) == 0:
        return 0.0

    learning_labels = map_labels_to_learning_labels(semantic_labels)

    valid_mask = np.ones(len(learning_labels), dtype=bool)
    for label_id, should_ignore in LEARNING_IGNORE.items():
        if should_ignore:
            valid_mask &= (learning_labels != label_id)

    learning_labels = learning_labels[valid_mask]

    if len(learning_labels) == 0:
        return 0.0

    label_counts = Counter(learning_labels)
    total_points = len(learning_labels)

    gini_sum = 0.0
    for label, count in label_counts.items():
        p_s = count / total_points
        gini_sum += p_s ** 2

    gdiv = 1.0 - gini_sum

    return gdiv

def compute_semantic_statistics(semantic_labels: np.ndarray,
                               ignore_unlabeled: bool = True) -> Dict:

    if semantic_labels is None or len(semantic_labels) == 0:
        return {
            "total_points": 0,
            "num_classes": 0,
            "class_distribution": {},
            "gini_diversity": 0.0
        }

    learning_labels = map_labels_to_learning_labels(semantic_labels)

    valid_mask = np.ones(len(learning_labels), dtype=bool)
    for label_id, should_ignore in LEARNING_IGNORE.items():
        if should_ignore:
            valid_mask &= (learning_labels != label_id)

    learning_labels = learning_labels[valid_mask]

    if len(learning_labels) == 0:
        return {
            "total_points": 0,
            "num_classes": 0,
            "class_distribution": {},
            "gini_diversity": 0.0
        }

    label_counts = Counter(learning_labels)
    total_points = len(learning_labels)

    class_distribution = {}
    for label, count in label_counts.items():

        if not LEARNING_IGNORE.get(label, False):
            class_name = LEARNING_LABELS.get(label, f"unknown_{label}")
            class_distribution[class_name] = {
                "count": int(count),
                "ratio": float(count / total_points)
            }

    gdiv = compute_gini_diversity(semantic_labels, ignore_unlabeled)

    return {
        "total_points": int(total_points),
        "num_classes": len(class_distribution),
        "class_distribution": class_distribution,
        "gini_diversity": float(gdiv)
    }

def load_semantic_labels(dataset, bg_index: int) -> Optional[np.ndarray]:

    try:
        if hasattr(dataset, 'get_semantic_label_3d'):
            semantic_labels = dataset.get_semantic_label_3d()
            return semantic_labels
        else:
            print("Warning: dataset does not provide semantic labels")
            return None
    except Exception as e:
        print(f"Error loading semantic labels for frame {bg_index}: {e}")
        return None

def get_category_prefix(group_name: str) -> str:

    import re

    prefix = re.sub(r'_\d+$', '', group_name)
    return prefix.lower()

def select_insertion_groups_by_diversity(
    available_groups: List[str],
    groups_detail: Dict,
    num_attempts: int = 5,
    seed: Optional[int] = None
) -> List[str]:

    if seed is not None:
        random.seed(seed)

    group_semantic_labels = {}
    for group_name in available_groups:
        group_detail = groups_detail.get(group_name, {})
        members = group_detail.get("members", [])
        labels = []
        for member in members:
            semantic_label = member.get("semantic_label", 0)

            learning_label = LEARNING_MAP.get(semantic_label, 0)
            if not LEARNING_IGNORE.get(learning_label, True):
                labels.append(learning_label)
        group_semantic_labels[group_name] = labels

    best_groups = []
    best_diversity = 0.0

    for attempt in range(num_attempts):

        num_groups = random.randint(1, min(5, len(available_groups)))
        sampled_groups = random.sample(available_groups, num_groups)

        all_labels = []
        for group_name in sampled_groups:
            all_labels.extend(group_semantic_labels.get(group_name, []))

        if len(all_labels) > 0:
            label_counts = Counter(all_labels)
            total = len(all_labels)
            gini_sum = sum((count / total) ** 2 for count in label_counts.values())
            diversity = 1.0 - gini_sum

            if diversity > best_diversity:
                best_diversity = diversity
                best_groups = sampled_groups

    return best_groups if best_groups else available_groups[:1]

def _generate_scene_with_groups(
    bg_index: int,
    selected_groups: List[str],
    groups_detail: Dict,
    shapenet_loader,
    dataset,
    dataset_output,
    road_split,
    dataset_type: str,
    merge_surfaces: bool = False
) -> Tuple[bool, Optional[np.ndarray]]:

    try:

        obj_names = []
        relative_displacement_list = []
        relative_rotation_list = []
        semantic_label_list = []
        obj_mesh_paths = []
        group_members_indices = []

        for group_name in selected_groups:
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

        calib_info = dataset.get_calibration()

        road_pc_valid, non_road_pc = road_split.split_pcd_road(
            bg_index,
            dataset.get_point_cloud_path(),
            dataset.road_split_label_dir,
            dataset.road_split_label_dir,
            insert_location="road",
            semantic_label_dir=getattr(dataset, "semantic_label_dir", None),
        )

        sidewalk_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "sidewalk_split_label")
        sidewalk_pc_valid, sidewalk_non_road_pc = road_split.split_pcd_road(
            bg_index,
            dataset.get_point_cloud_path(),
            sidewalk_label_dir,
            sidewalk_label_dir,
            insert_location="sidewalk",
            semantic_label_dir=getattr(dataset, "semantic_label_dir", None),
        )
        if sidewalk_pc_valid is None:
            sidewalk_non_road_pc = None

        terrain_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "terrain_split_label")
        terrain_pc_valid, terrain_non_road_pc = road_split.split_pcd_road(
            bg_index,
            dataset.get_point_cloud_path(),
            terrain_label_dir,
            terrain_label_dir,
            insert_location="terrain",
            semantic_label_dir=getattr(dataset, "semantic_label_dir", None),
        )
        if terrain_pc_valid is None:
            terrain_non_road_pc = None

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
            lower = name.lower()
            for prefix, surface in SURFACE_PREFIX_MAP:
                if lower.startswith(prefix):
                    return surface
            return "road"

        if merge_surfaces:
            road_groups = []
            sidewalk_groups = []
            terrain_groups = []
            skipped_groups = []

            available_surfaces = []
            if road_pc_valid is not None:
                available_surfaces.append("road")
            if sidewalk_pc_valid is not None:
                available_surfaces.append("sidewalk")
            if terrain_pc_valid is not None:
                available_surfaces.append("terrain")

            if not available_surfaces:
                print("Warning: no surface point clouds are available; skipping all selected groups")
                skipped_groups = group_members_indices
            else:
                for member_indices in group_members_indices:
                    if not member_indices:
                        continue

                    first_member_name = obj_names[member_indices[0]]
                    correct_surface = category_of(first_member_name)

                    wrong_surfaces = [s for s in available_surfaces if s != correct_surface]

                    if wrong_surfaces:
                        assigned_surface = random.choice(wrong_surfaces)
                    else:
                        print(f"Warning: {first_member_name} skipped because only the correct surface type is available")
                        skipped_groups.append(member_indices)
                        continue

                    if assigned_surface == "road":
                        road_groups.append(member_indices)
                    elif assigned_surface == "sidewalk":
                        sidewalk_groups.append(member_indices)
                    elif assigned_surface == "terrain":
                        terrain_groups.append(member_indices)

        else:

            road_groups = []
            sidewalk_groups = []
            terrain_groups = []
            skipped_groups = []

            for member_indices in group_members_indices:
                if not member_indices:
                    continue
                first_member_name = obj_names[member_indices[0]]
                surface_type = category_of(first_member_name)

                if surface_type == "sidewalk":
                    if sidewalk_pc_valid is not None:
                        sidewalk_groups.append(member_indices)
                    else:
                        print(f"Warning: {first_member_name} requires sidewalk data; falling back to road surface")
                        road_groups.append(member_indices)

                elif surface_type == "terrain":
                    if terrain_pc_valid is not None:
                        terrain_groups.append(member_indices)
                    else:
                        print(f"Warning: {first_member_name} skipped because terrain data is unavailable")
                        skipped_groups.append(member_indices)

                else:
                    road_groups.append(member_indices)

            if skipped_groups:
                skipped_count = sum(len(g) for g in skipped_groups)
                print(f"Warning: skipped {len(skipped_groups)} group(s) ({skipped_count} objects) due to missing surface data")

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

        success_indices = [i for i, p in enumerate(obj_lidar_positions) if p is not None]
        if not success_indices:
            print("Warning: no objects were inserted successfully")
            return False, None

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

        if hasattr(dataset_output, 'semantic_label_3d') and dataset_output.semantic_label_3d is not None:
            return True, dataset_output.semantic_label_3d
        else:

            original_labels = load_semantic_labels(dataset, bg_index)
            if original_labels is not None:
                estimated_labels = list(original_labels)
                for sem_label in semantic_label_list:
                    estimated_labels.extend([sem_label] * 1000)
                return True, np.array(estimated_labels, dtype=np.uint32)
            else:
                return True, None

    except Exception as e:
        print(f"Error generating scene: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def analyze_original_diversity(bg_index: int) -> Dict:

    container = DataGenContainer()
    dataset_config = container.dataset_config()
    dataset_type = dataset_config.get_config("source_dataset").get("dataset_type", "")
    source_dataset_cfg = dataset_config.get_config("source_dataset")

    dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, source_dataset_cfg)

    semantic_labels = load_semantic_labels(dataset, bg_index)

    if semantic_labels is None:
        print(f"Warning: frame {bg_index} has no semantic labels")
        return {
            "bg_index": bg_index,
            "gini_diversity": 0.0,
            "num_classes": 0,
            "total_points": 0,
            "class_distribution": {}
        }

    stats = compute_semantic_statistics(semantic_labels)
    stats["bg_index"] = bg_index

    print(
        f"Baseline frame {bg_index}: gini {stats['gini_diversity']:.4f}, "
        f"classes {stats['num_classes']}, points {stats['total_points']}"
    )

    return stats

def generate_with_diversity_guidance(
    bg_index: int,
    category_pool: Optional[List[str]] = None,
    required_categories: Optional[List[str]] = None,
    max_attempts: int = 20,
    selection_mode: str = "first_better",
    merge_surfaces: bool = False
) -> Dict:

    container = DataGenContainer()
    road_split = container.road_split()
    shapenet_loader = container.shapenet_loader()
    dataset_config = container.dataset_config()

    dataset_type = dataset_config.get_config("source_dataset").get("dataset_type", "")
    dataset_output_type = dataset_config.get_config("source_dataset").get("dataset_output_type", "")
    source_dataset_cfg = dataset_config.get_config("source_dataset")

    obj_insert_config_rel_path = source_dataset_cfg.get("obj_insert_config_path", "configs/obj_insert_config.yml")
    if not os.path.isabs(obj_insert_config_rel_path):
        obj_insert_config_path = PROJECT_ROOT / obj_insert_config_rel_path
    else:
        obj_insert_config_path = Path(obj_insert_config_rel_path)

    with open(obj_insert_config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    config_file = yaml_config.get("config_file", "obj_insert_config.json")
    if not os.path.isabs(config_file):
        json_config_path = obj_insert_config_path.parent / config_file
    else:
        json_config_path = Path(config_file)

    with open(json_config_path, "r") as f:
        json_config = json.load(f)

    groups_detail = json_config.get("groups", {})

    if category_pool is not None:
        available_groups = []
        for cat in category_pool:
            matching_groups = [g for g in groups_detail.keys() if g.lower().startswith(cat.lower())]
            available_groups.extend(matching_groups)
        if not available_groups:
            print(f"Warning: no groups matched {category_pool}; using all available groups")
            available_groups = list(groups_detail.keys())
    else:
        available_groups = list(groups_detail.keys())

    required_groups = []
    if required_categories is not None:
        for cat in required_categories:
            matching_groups = [g for g in groups_detail.keys() if g.lower().startswith(cat.lower())]
            required_groups.extend(matching_groups)
        if not required_groups:
            print(f"Warning: no groups matched required categories {required_categories}")

    print(f"\nFrame {bg_index}: {len(available_groups)} candidate groups")
    if required_groups and selection_mode == "best_of_all":
        print(f"Required category groups queued: {len(required_groups)}")

    dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, source_dataset_cfg)
    dataset_output = DataFrameFactory.get_output_data_frame(
        dataset_type, dataset_output_type, bg_index, dataset_config.get_config("generated_dataset")
    )

    NUM_GROUPS_PER_ATTEMPT = 1

    original_semantic_labels = load_semantic_labels(dataset, bg_index)
    if original_semantic_labels is not None:
        original_gdiv = compute_gini_diversity(original_semantic_labels)
        print(f"Baseline Gini diversity: {original_gdiv:.4f}")
    else:
        original_gdiv = 0.0
        print("Warning: unable to load baseline semantic labels")

    tried_category_prefixes = set()

    if selection_mode == "first_better":

        last_config = None
        found_better = False
    elif selection_mode == "best_of_all":

        all_configs = []
        best_config = None
        best_improvement = -float('inf')

        tried_required_groups = set()
        remaining_required_groups = required_groups.copy() if required_groups else []
    elif selection_mode == "no_guidance":

        pass
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    last_success_config = None

    for attempt in range(max_attempts):

        if selection_mode == "no_guidance":
            if len(available_groups) < NUM_GROUPS_PER_ATTEMPT:
                selected_groups = available_groups.copy()
            else:
                selected_groups = random.sample(available_groups, NUM_GROUPS_PER_ATTEMPT)
        else:

            untried_available_groups = [
                g for g in available_groups
                if get_category_prefix(g) not in tried_category_prefixes
            ]

            if not untried_available_groups:
                untried_available_groups = available_groups

            if selection_mode == "best_of_all" and remaining_required_groups:

                untried_required = [g for g in remaining_required_groups
                                   if get_category_prefix(g) not in tried_category_prefixes]

                if untried_required:
                    selected_required = random.choice(untried_required)
                else:

                    selected_required = random.choice(remaining_required_groups)

                selected_groups = [selected_required]

                remaining_required_groups.remove(selected_required)
                tried_required_groups.add(selected_required)

                remaining_count = NUM_GROUPS_PER_ATTEMPT - 1
                if remaining_count > 0:

                    other_groups = [g for g in untried_available_groups if g not in selected_groups]
                    if other_groups:
                        if len(other_groups) < remaining_count:
                            selected_groups.extend(other_groups)
                        else:
                            selected_groups.extend(random.sample(other_groups, remaining_count))
            else:

                if len(untried_available_groups) < NUM_GROUPS_PER_ATTEMPT:
                    selected_groups = untried_available_groups.copy()
                else:
                    selected_groups = random.sample(untried_available_groups, NUM_GROUPS_PER_ATTEMPT)

        success, generated_labels = _generate_scene_with_groups(
            bg_index, selected_groups, groups_detail,
            shapenet_loader, dataset, dataset_output, road_split, dataset_type,
            merge_surfaces=merge_surfaces
        )

        if not success or generated_labels is None:
            print("Warning: scene generation failed for current selection; skipping attempt")
            continue

        if selection_mode != "no_guidance":
            for g in selected_groups:
                tried_category_prefixes.add(get_category_prefix(g))

        current_gdiv = compute_gini_diversity(generated_labels)
        current_improvement = current_gdiv - original_gdiv

        current_config = {
            "groups": selected_groups,
            "gdiv": current_gdiv,
            "semantic_labels": generated_labels,
            "attempt": attempt + 1,
            "improvement": current_improvement
        }
        last_success_config = current_config

        if selection_mode == "first_better":

            last_config = current_config

            if current_gdiv > original_gdiv:
                found_better = True
                break

        elif selection_mode == "best_of_all":

            all_configs.append(current_config)

            if current_improvement > best_improvement:
                best_improvement = current_improvement
                best_config = current_config

        elif selection_mode == "no_guidance":
            final_config = current_config
            break

    if selection_mode == "first_better":

        final_config = last_config

        if final_config is not None:
            status = "improved over baseline" if found_better else "no improvement after max attempts; using last result"
            print(f"\nFirst-better result: {status}. Final Gini {final_config['gdiv']:.4f} (baseline {original_gdiv:.4f}).")

            final_stats = compute_semantic_statistics(final_config['semantic_labels'])

            return {
                "bg_index": bg_index,
                "original_gdiv": original_gdiv,
                "final_gdiv": final_config['gdiv'],
                "improvement": final_config['improvement'],
                "groups": final_config['groups'],
                "num_classes": final_stats['num_classes'],
                "total_points": final_stats['total_points'],
                "class_distribution": final_stats['class_distribution'],
                "found_better": found_better,
                "attempt": final_config['attempt'],
                "selection_mode": selection_mode,
                "success": True
            }

    elif selection_mode == "best_of_all":

        final_config = best_config

        if final_config is not None:
            need_regenerate = (
                last_success_config is None
                or final_config['attempt'] != last_success_config['attempt']
            )
            if need_regenerate:
                print("Info: regenerating the best configuration to refresh saved outputs")
                dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, source_dataset_cfg)
                dataset_output = DataFrameFactory.get_output_data_frame(
                    dataset_type, dataset_output_type, bg_index, dataset_config.get_config("generated_dataset")
                )
                regen_success, regen_labels = _generate_scene_with_groups(
                    bg_index,
                    final_config['groups'],
                    groups_detail,
                    shapenet_loader,
                    dataset,
                    dataset_output,
                    road_split,
                    dataset_type,
                    merge_surfaces=merge_surfaces,
                )
                if regen_success and regen_labels is not None:
                    regen_gdiv = compute_gini_diversity(regen_labels)
                    final_config = {
                        **final_config,
                        "gdiv": regen_gdiv,
                        "improvement": regen_gdiv - original_gdiv,
                        "semantic_labels": regen_labels,
                    }
                    best_config = final_config
                    best_improvement = final_config["improvement"]
                else:
                    print("Warning: unable to regenerate the best configuration; keeping previous metrics")

            all_improvements = [cfg['improvement'] for cfg in all_configs]
            final_stats = compute_semantic_statistics(final_config['semantic_labels'])

            return {
                "bg_index": bg_index,
                "original_gdiv": original_gdiv,
                "final_gdiv": final_config['gdiv'],
                "improvement": final_config['improvement'],
                "groups": final_config['groups'],
                "num_classes": final_stats['num_classes'],
                "total_points": final_stats['total_points'],
                "class_distribution": final_stats['class_distribution'],
                "attempt": final_config['attempt'],
                "selection_mode": selection_mode,
                "all_improvements": all_improvements,
                "success": True
            }

    elif selection_mode == "no_guidance":

        final_config = locals().get('final_config', None)

        if final_config is not None:
            final_stats = compute_semantic_statistics(final_config['semantic_labels'])

            return {
                "bg_index": bg_index,
                "original_gdiv": original_gdiv,
                "final_gdiv": final_config['gdiv'],
                "improvement": final_config['improvement'],
                "groups": final_config['groups'],
                "num_classes": final_stats['num_classes'],
                "total_points": final_stats['total_points'],
                "class_distribution": final_stats['class_distribution'],
                "attempt": final_config['attempt'],
                "selection_mode": selection_mode,
                "success": True
            }

    error_msg = f"All {max_attempts} attempts failed; no objects were inserted"
    print(f"\n{'='*60}")
    print(f"Error: {error_msg}")
    print(f"{'='*60}")

    raise RuntimeError(error_msg)

if __name__ == '__main__':
    print("===== RQ2: Semantic Diversity Guidance =====\n")

    bgs_override = os.environ.get("SELDA_RQ2_BGS_FILE")
    if bgs_override:
        bgs_path = Path(bgs_override)
        if not bgs_path.is_absolute():
            bgs_path = TOOLS_DIR / bgs_path
    else:
        bgs_path = DEFAULT_BGS_FILE
    error_path = ERROR_FILE

    indices = []
    try:
        if bgs_path.exists():
            with open(bgs_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        v = int(s)
                        if 0 <= v <= MAX_BG_INDEX:
                            indices.append(v)
                    except Exception:
                        continue

            indices = indices[:500]
        else:

            upper = MAX_BG_INDEX + 1
            k = min(200, upper)
            indices = sorted(random.sample(range(0, upper), k=k))
            with open(bgs_path, 'w', encoding='utf-8') as f:
                for v in indices:
                    f.write(f"{v}\n")
    except Exception as e:
        print(f"Failed preparing bg indices: {e}")
        indices = [0]

    print(f"Total frames to process: {len(indices)}\n")

    print("=" * 60)
    print("Step 1: analyze baseline scenes (no insertion)")
    print("=" * 60)

    original_results = []
    for ix, bg_index in enumerate(indices, start=1):
        try:
            print(f"\nAnalyzing frame {ix}/{len(indices)} (bg_index={bg_index})")
            stats = analyze_original_diversity(bg_index)
            original_results.append(stats)
        except Exception as e:
            print(f"Error at frame {bg_index}: {e}")
            continue

    with open(ORIGINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(original_results, f, indent=2)

    if original_results:
        gdiv_values = [r["gini_diversity"] for r in original_results if r["gini_diversity"] > 0]
        if gdiv_values:
            print(f"\n{'='*60}")
            print("Baseline diversity statistics:")
            print(f"  Mean Gini diversity: {np.mean(gdiv_values):.4f} ± {np.std(gdiv_values):.4f}")
            print(f"  Minimum: {np.min(gdiv_values):.4f}")
            print(f"  Maximum: {np.max(gdiv_values):.4f}")
            print(f"  Saved to: {ORIGINAL_OUTPUT_FILE}")
            print(f"{'='*60}\n")

    print("\n" + "=" * 60)
    print("Step 2: diversity-guided generation")
    print("=" * 60)

    # TODO best_of_all / no_guidance
    SELECTION_MODE = "best_of_all"

    # TODO
    MERGE_SURFACES = False

    category_pool = []

    required_categories = []

    print(f"Selection mode: {SELECTION_MODE}")
    print(f"Surface merge enabled: {MERGE_SURFACES}")
    print(f"Category pool: {category_pool}")
    print(f"Required categories: {required_categories}\n")

    generated_results = []
    for ix, bg_index in enumerate(indices, start=1):
        try:
            print(f"\nProcessing frame {ix}/{len(indices)} (bg_index={bg_index})")

            result = generate_with_diversity_guidance(
                bg_index=bg_index,
                category_pool=category_pool,
                required_categories=required_categories,
                max_attempts=5,
                selection_mode=SELECTION_MODE,
                merge_surfaces=MERGE_SURFACES
            )

            generated_results.append(result)

        except Exception as e:
            print(f"\nError while processing frame {bg_index}: {e}")
            import traceback
            traceback.print_exc()

            try:
                with open(error_path, 'a', encoding='utf-8') as ef:
                    ef.write(f"{bg_index}\n")
                print(f"Recorded failed frame in {error_path}")
            except Exception as write_error:
                print(f"Warning: unable to write to error log: {write_error}")

            continue

    with open(GENERATED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(generated_results, f, indent=2)

    if generated_results:
        success_results = [r for r in generated_results if r.get("success", True)]
        if success_results:
            improvements = [r["improvement"] for r in success_results]
            final_gdivs = [r["final_gdiv"] for r in success_results]

            print("\n" + "=" * 60)
            print("Summary of generated frames:")
            print("=" * 60)
            print(f"Total frames: {len(indices)}")
            print(f"Succeeded: {len(success_results)}")
            print(f"Failed: {len(indices) - len(success_results)}")
            print(f"\nAvg. Gini improvement: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
            print(f"Min improvement: {np.min(improvements):.4f}")
            print(f"Max improvement: {np.max(improvements):.4f}")
            print(f"\nAvg. final Gini: {np.mean(final_gdivs):.4f} ± {np.std(final_gdivs):.4f}")
            print("\nResults saved to:")
            print(f"  Baseline stats: {ORIGINAL_OUTPUT_FILE}")
            print(f"  Generated stats: {GENERATED_OUTPUT_FILE}")
            if error_path.exists():
                print(f"  Error log: {error_path}")
            print("=" * 60 + "\n")

    print("RQ2 pipeline finished")

