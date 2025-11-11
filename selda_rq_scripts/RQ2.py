#!/usr/bin/env python3
"""
RQ2: 语义多样性引导策略 (Semantic Diversity Guidance)
实现基于基尼不纯度的语义多样性度量和引导数据生成
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

def _locate_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current] + list(current.parents):
        if (candidate / "entry" / "containers.py").exists():
            return candidate
    raise RuntimeError("Unable to locate Selda project root (missing entry/containers.py)")


PROJECT_ROOT = _locate_project_root()

os.environ.setdefault("DRIVENCE_ROOT_DIR", str(PROJECT_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.containers import DataGenContainer
from drivence.assets.assets_utils import get_obj_path_for_member
from drivence.dataset.common.dataframe_factory import DataFrameFactory
from drivence.entity.scene_info import SceneInfo
from drivence.module.data_gen import main
from drivence.module.logic_scene.logic_scene_generator import create_logic_scene_generator
from drivence.module.pose_gen.pose_generation import generate_group_pose_from_road
from drivence.utils.common_utils import UUIDManager
from drivence.utils.path_utils import get_project_root_dir


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_BGS_FILE = TOOLS_DIR / "bgs_rq2_0.txt"
ERROR_FILE = TOOLS_DIR / "error_rq2.txt"
ORIGINAL_OUTPUT_FILE = TOOLS_DIR / "rq2_original_diversity.json"
GENERATED_OUTPUT_FILE = TOOLS_DIR / "rq2_generated_diversity.json"
MAX_BG_INDEX = 4070

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ========== 语义标签映射规则（基于 SemanticKITTI） ==========
# 参考: /home/atri/WD_Passport/CENet/CENet/config/labels/semantic-kitti.yaml

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

# 学习映射：将原始标签映射到学习标签 (0-20)
# 特殊规则：
#   - 99 号标签 (other-object) → 计入 Gini 计算 ✅
#   - 98 号标签 (custom-object) → 计入 Gini 计算（当作 99 号处理）✅
LEARNING_MAP = {
    0: 0,     # "unlabeled"
    1: 0,     # "outlier" -> "unlabeled"
    10: 1,    # "car"
    11: 2,    # "bicycle"
    13: 5,    # "bus" -> "other-vehicle"
    15: 3,    # "motorcycle"
    16: 5,    # "on-rails" -> "other-vehicle"
    18: 4,    # "truck"
    20: 5,    # "other-vehicle"
    30: 6,    # "person"
    31: 7,    # "bicyclist"
    32: 8,    # "motorcyclist"
    40: 9,    # "road"
    44: 10,   # "parking"
    48: 11,   # "sidewalk"
    49: 12,   # "other-ground"
    50: 13,   # "building"
    51: 14,   # "fence"
    52: 0,    # "other-structure" -> "unlabeled"
    60: 9,    # "lane-marking" -> "road"
    70: 15,   # "vegetation"
    71: 16,   # "trunk"
    72: 17,   # "terrain"
    80: 18,   # "pole"
    81: 19,   # "traffic-sign"
    98: 20,   # "custom-object" -> "other-object" (计入计算) ✅
    99: 20,   # "other-object" (计入计算) ✅
    252: 1,   # "moving-car" -> "car"
    253: 7,   # "moving-bicyclist" -> "bicyclist"
    254: 6,   # "moving-person" -> "person"
    255: 8,   # "moving-motorcyclist" -> "motorcyclist"
    256: 5,   # "moving-on-rails" -> "other-vehicle"
    257: 5,   # "moving-bus" -> "other-vehicle"
    258: 4,   # "moving-truck" -> "truck"
    259: 5,   # "moving-other-vehicle" -> "other-vehicle"
}

# 学习标签名称 (0-20)
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
    20: "other-object"  # ✅ 98号和99号标签都映射到这里
}

# 忽略的类别
# 排除静态场景类别，只关注动态物体和重要元素
LEARNING_IGNORE = {
    0: True,   # "unlabeled" 应被忽略
    1: False,  # "car" 计入 ✓
    2: False,  # "bicycle" 计入 ✓
    3: False,  # "motorcycle" 计入 ✓
    4: False,  # "truck" 计入 ✓
    5: False,  # "other-vehicle" 计入 ✓
    6: False,  # "person" 计入 ✓
    7: False,  # "bicyclist" 计入 ✓
    8: False,  # "motorcyclist" 计入 ✓
    9: True,   # "road" 不计入 ❌
    10: True,  # "parking" 不计入 ❌
    11: True,  # "sidewalk" 不计入 ❌
    12: True,  # "other-ground" 不计入 ❌
    13: True,  # "building" 不计入 ❌
    14: False, # "fence" 计入 ✓
    15: False, # "vegetation" 计入 ✓
    16: False, # "trunk" 计入 ✓
    17: True,  # "terrain" 不计入 ❌
    18: False, # "pole" 计入 ✓
    19: False, # "traffic-sign" 计入 ✓
    20: False  # "other-object" (98号和99号) 计入 ✓
}


# ========== 语义多样性度量：基尼不纯度（Gini Diversity） ==========

def map_labels_to_learning_labels(semantic_labels: np.ndarray) -> np.ndarray:
    """
    将原始语义标签映射到学习标签
    
    Args:
        semantic_labels: 原始语义标签数组（uint32或uint16）
        
    Returns:
        学习标签数组
    """
    # 提取语义标签（对于 SemanticKITTI 的 .label 文件，取低16位）
    if semantic_labels.dtype == np.uint32:
        semantic_labels = semantic_labels & 0xFFFF
    
    # 映射到学习标签
    learning_labels = np.zeros_like(semantic_labels, dtype=np.uint8)
    for orig_label, learn_label in LEARNING_MAP.items():
        learning_labels[semantic_labels == orig_label] = learn_label
    
    return learning_labels


def compute_gini_diversity(semantic_labels: np.ndarray, 
                          ignore_unlabeled: bool = True) -> float:
    """
    计算语义多样性度量：基尼不纯度 (Gini Diversity)
    
    公式: Gdiv = 1 - Σ(p_s^2)
    其中 p_s 是类别 s 的点数占总点数的比例
    
    Args:
        semantic_labels: 语义标签数组（已映射到学习标签 0-20）
        ignore_unlabeled: 是否忽略 unlabeled (label 0) 类别
        
    Returns:
        基尼多样性值 [0, 1]，值越高表示多样性越强
        
    注意：
        - 99 号标签 (other-object) 计入 Gini 计算 ✅
        - 98 号标签 (custom-object) 计入 Gini 计算（当作 99 号处理）✅
        
    排除的静态场景类别（不计入 Gini）：
        ❌ road (9)
        ❌ parking (10)
        ❌ sidewalk (11)
        ❌ other-ground (12)
        ❌ building (13)
        ❌ terrain (17)
        ❌ other-structure (52 → 0)
        
    计入的动态/重要类别：
        ✓ 所有车辆类别 (car, truck, bicycle, motorcycle, other-vehicle)
        ✓ 所有人类相关 (person, bicyclist, motorcyclist)
        ✓ 植被 (vegetation, trunk)
        ✓ 基础设施 (fence, pole, traffic-sign)
        ✓ 其他物体 (other-object, 包括98号和99号标签)
    """
    if semantic_labels is None or len(semantic_labels) == 0:
        return 0.0
    
    # 映射到学习标签
    learning_labels = map_labels_to_learning_labels(semantic_labels)
    
    # 过滤掉所有被忽略的类别（使用 LEARNING_IGNORE 配置）
    valid_mask = np.ones(len(learning_labels), dtype=bool)
    for label_id, should_ignore in LEARNING_IGNORE.items():
        if should_ignore:
            valid_mask &= (learning_labels != label_id)
    
    learning_labels = learning_labels[valid_mask]
    
    if len(learning_labels) == 0:
        return 0.0
    
    # 统计每个类别的点数
    label_counts = Counter(learning_labels)
    total_points = len(learning_labels)
    
    # 计算基尼不纯度
    gini_sum = 0.0
    for label, count in label_counts.items():
        p_s = count / total_points
        gini_sum += p_s ** 2
    
    gdiv = 1.0 - gini_sum
    
    return gdiv


def compute_semantic_statistics(semantic_labels: np.ndarray,
                               ignore_unlabeled: bool = True) -> Dict:
    """
    计算语义统计信息
    
    Args:
        semantic_labels: 语义标签数组
        ignore_unlabeled: 是否忽略 unlabeled 类别
        
    Returns:
        统计信息字典
    """
    if semantic_labels is None or len(semantic_labels) == 0:
        return {
            "total_points": 0,
            "num_classes": 0,
            "class_distribution": {},
            "gini_diversity": 0.0
        }
    
    # 映射到学习标签
    learning_labels = map_labels_to_learning_labels(semantic_labels)
    
    # 过滤掉所有被忽略的类别（包括 unlabeled 和静态场景类别）
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
    
    # 统计每个类别的点数
    label_counts = Counter(learning_labels)
    total_points = len(learning_labels)
    
    # 计算类别分布（只显示未被忽略的类别）
    class_distribution = {}
    for label, count in label_counts.items():
        # 双重检查：确保不显示被忽略的类别
        if not LEARNING_IGNORE.get(label, False):
            class_name = LEARNING_LABELS.get(label, f"unknown_{label}")
            class_distribution[class_name] = {
                "count": int(count),
                "ratio": float(count / total_points)
            }
    
    # 计算基尼多样性
    gdiv = compute_gini_diversity(semantic_labels, ignore_unlabeled)
    
    return {
        "total_points": int(total_points),
        "num_classes": len(class_distribution),  # 只计数未被忽略的类别
        "class_distribution": class_distribution,
        "gini_diversity": float(gdiv)
    }


def load_semantic_labels(dataset, bg_index: int) -> Optional[np.ndarray]:
    """
    加载指定帧的语义标签
    
    Args:
        dataset: 数据集对象
        bg_index: 背景帧索引
        
    Returns:
        语义标签数组，如果不存在则返回 None
    """
    try:
        if hasattr(dataset, 'get_semantic_label_3d'):
            semantic_labels = dataset.get_semantic_label_3d()
            return semantic_labels
        else:
            print(f"Warning: 数据集不支持语义标签")
            return None
    except Exception as e:
        print(f"Error loading semantic labels for frame {bg_index}: {e}")
        return None


# ========== 路面分割：基于最近点距离 ==========

def split_road_by_distance(
    bg_index: int,
    bg_pc_path: str,
    road_split_label_dir: str,
    calib_info,
    distance_threshold: float = 1.0,
    img_height: int = 375,
    img_width: int = 1242
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于最近点距离的路面分割（避免索引越界问题）
    
    工作原理：
    1. 加载预生成的路面点云（.bin 文件）
    2. 加载当前背景点云
    3. 使用 KD-Tree 计算每个背景点到路面点云的最近距离
    4. 距离 < threshold 的点判定为路面点
    5. 过滤出在图像视野内的路面点
    
    Args:
        bg_index: 背景帧索引
        bg_pc_path: 背景点云路径
        road_split_label_dir: 路面分割标签目录
        calib_info: 标定信息
        distance_threshold: 距离阈值（米）
        img_height: 图像高度
        img_width: 图像宽度
        
    Returns:
        (road_pc_valid, non_road_pc) 元组
    """
    from scipy.spatial import cKDTree
    
    # 加载背景点云
    pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    
    # 加载预生成的路面点云
    road_interpolation_path = os.path.join(road_split_label_dir, f"{bg_index:06d}.bin")
    
    if not os.path.exists(road_interpolation_path):
        print(f"Warning: 路面点云文件不存在: {road_interpolation_path}")
        print(f"使用原始方法进行路面分割...")
        # 这里应该调用原始的 road_split 方法，但为了简化暂时返回 None
        return None, None
    
    # 加载路面点云
    pc_road = np.fromfile(road_interpolation_path, dtype=np.float32).reshape(-1, 3)
    
    if len(pc_road) == 0:
        print(f"Warning: 路面点云为空")
        return None, None
    
    print(f"路面点云数量: {len(pc_road)}, 背景点云数量: {len(pc_bg)}")
    
    # 构建路面点云的 KD-Tree
    road_tree = cKDTree(pc_road)
    
    # 批量查询每个背景点到路面的最近距离
    distances, _ = road_tree.query(pc_bg, k=1)
    
    # 基于距离阈值判定路面点
    road_mask = distances < distance_threshold
    non_road_mask = ~road_mask
    
    # 分离路面点和非路面点
    pc_bg_road = pc_bg[road_mask]
    pc_bg_non_road = pc_bg[non_road_mask]
    
    print(f"基于距离分割结果: 路面点={len(pc_bg_road)}, 非路面点={len(pc_bg_non_road)}")
    
    # 过滤出在图像视野内的路面点
    pts_img, pts_rect_depth = calib_info.lidar_to_img(pc_bg_road)
    
    img_shape = (img_height, img_width)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    road_pc_valid = pc_bg_road[pts_valid_flag]
    
    print(f"图像视野内的路面点: {len(road_pc_valid)}")
    
    return road_pc_valid, pc_bg_non_road


# ========== 语义多样性引导策略 ==========

def get_category_prefix(group_name: str) -> str:
    """
    提取组名的类别前缀（去掉末尾的下划线和数字/字母组合）
    
    示例:
        car_1 -> car
        truck_2 -> truck
        person-riding-bicycle_3 -> person-riding-bicycle
        Ambulance_1 -> ambulance
        adult_1 -> adult
        tree -> tree (没有后缀)
    
    Args:
        group_name: 组名
        
    Returns:
        类别前缀（小写）
    """
    import re
    # 移除末尾的 _数字 (如 _1, _2, _123)
    # 注意：只移除下划线后紧跟数字的情况
    prefix = re.sub(r'_\d+$', '', group_name)
    return prefix.lower()


def select_insertion_groups_by_diversity(
    available_groups: List[str],
    groups_detail: Dict,
    num_attempts: int = 5,
    seed: Optional[int] = None
) -> List[str]:
    """
    基于语义多样性贪心选择插入的组
    
    算法：
    1. 对每个可用组，估计其会引入的语义类别
    2. 尝试多次随机采样，每次计算采样后的预期语义多样性
    3. 选择能最大化多样性的组合
    
    Args:
        available_groups: 可用的组名列表
        groups_detail: 组的详细信息（包含 semantic_label）
        num_attempts: 尝试次数
        seed: 随机种子
        
    Returns:
        选中的组名列表
    """
    if seed is not None:
        random.seed(seed)
    
    # 提取每个组的语义类别
    group_semantic_labels = {}
    for group_name in available_groups:
        group_detail = groups_detail.get(group_name, {})
        members = group_detail.get("members", [])
        labels = []
        for member in members:
            semantic_label = member.get("semantic_label", 0)
            # 映射到学习标签
            learning_label = LEARNING_MAP.get(semantic_label, 0)
            if not LEARNING_IGNORE.get(learning_label, True):
                labels.append(learning_label)
        group_semantic_labels[group_name] = labels
    
    best_groups = []
    best_diversity = 0.0
    
    # 多次尝试
    for attempt in range(num_attempts):
        # 随机采样组
        num_groups = random.randint(1, min(5, len(available_groups)))
        sampled_groups = random.sample(available_groups, num_groups)
        
        # 收集所有语义标签
        all_labels = []
        for group_name in sampled_groups:
            all_labels.extend(group_semantic_labels.get(group_name, []))
        
        # 计算预期多样性（基于语义标签的分布）
        if len(all_labels) > 0:
            label_counts = Counter(all_labels)
            total = len(all_labels)
            gini_sum = sum((count / total) ** 2 for count in label_counts.values())
            diversity = 1.0 - gini_sum
            
            # 贪心更新
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
    modality: str = "pc",
    merge_surfaces: bool = False
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    使用选定的组生成场景并返回语义标签
    
    Args:
        bg_index: 背景帧索引
        selected_groups: 选中的组名列表
        groups_detail: 组详细信息
        shapenet_loader: ShapeNet 加载器
        dataset: 数据集对象
        dataset_output: 输出数据集对象
        road_split: 道路分割对象
        dataset_type: 数据集类型
        modality: 模态
        merge_surfaces: 是否打乱物体与地面的对应关系（True=打乱，False=正常）
                       - False: 物体按类别插入对应地面（car→road, person→sidewalk, tree→terrain）
                       - True: 💥 故意打乱对应关系，随机分配物体到错误的地面
                              （例如：car可能插入到sidewalk，person可能插入到road）
                              目的：引发更多碰撞和错误场景，用于压力测试
        
    Returns:
        (success, semantic_labels) 元组
    """
    try:
        # 准备插入配置
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
        
        # 加载 mesh
        mesh_list = []
        scale_ratio_arr = []
        objs_index_arr = []
        
        for i, obj_name in enumerate(obj_names):
            # 检查是否有帧号配置
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
        
        # 获取道路分割和标定信息
        calib_info = dataset.get_calibration()
        
        # 使用改进的路面分割：基于最近点距离判断，避免索引越界
        road_pc_valid, non_road_pc = split_road_by_distance(
            bg_index,
            dataset.get_point_cloud_path(),
            dataset.road_split_label_dir,
            calib_info,
            distance_threshold=0.3  # 距离阈值（米）
        )
        
        # 生成 sidewalk 和 terrain 点云（如果存在）
        sidewalk_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "sidewalk_split_label")
        sidewalk_pc_valid = None
        sidewalk_non_road_pc = None
        if os.path.isdir(sidewalk_label_dir):
            sidewalk_pc_valid, sidewalk_non_road_pc = split_road_by_distance(
                bg_index,
                dataset.get_point_cloud_path(),
                sidewalk_label_dir,
                calib_info,
                distance_threshold=0.3
            )
            # 如果失败，设置为 None
            if sidewalk_pc_valid is None:
                sidewalk_non_road_pc = None
        
        terrain_label_dir = os.path.join(os.path.dirname(dataset.road_split_label_dir), "terrain_split_label")
        terrain_pc_valid = None
        terrain_non_road_pc = None
        if os.path.isdir(terrain_label_dir):
            terrain_pc_valid, terrain_non_road_pc = split_road_by_distance(
                bg_index,
                dataset.get_point_cloud_path(),
                terrain_label_dir,
                calib_info,
                distance_threshold=0.3
            )
            # 如果失败，设置为 None
            if terrain_pc_valid is None:
                terrain_non_road_pc = None
        
        # 类别到地面类型映射（与 RQ1.py 相同）
        SURFACE_PREFIX_MAP = [
            # ("bicycle", "sidewalk"),
            # ("motorcycle", "sidewalk"),
            # ("motor", "sidewalk"),
            ("scooter", "sidewalk"),
            ("person", "sidewalk"),
            # ("bicyclist", "sidewalk"),
            # ("motorcyclist", "sidewalk"),
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
        
        # 根据 merge_surfaces 参数决定地面分配策略
        if merge_surfaces:
            # 🔥 打乱模式：故意将物体插入到错误的地面上，引发更多碰撞和错误
            print(f"💥 打乱地面模式：随机分配物体到错误的地面类型")
            
            road_groups = []
            sidewalk_groups = []
            terrain_groups = []
            skipped_groups = []
            
            # 构建可用地面列表（只包括存在的地面）
            available_surfaces = []
            if road_pc_valid is not None:
                available_surfaces.append("road")
            if sidewalk_pc_valid is not None:
                available_surfaces.append("sidewalk")
            if terrain_pc_valid is not None:
                available_surfaces.append("terrain")
            
            if not available_surfaces:
                print(f"❌ 错误：没有任何可用的地面点云")
                skipped_groups = group_members_indices
            else:
                print(f"  可用地面类型: {available_surfaces}")
                
                # 对每个组，强制分配到错误的地面类型（100%错误）
                for member_indices in group_members_indices:
                    if not member_indices:
                        continue
                    
                    first_member_name = obj_names[member_indices[0]]
                    correct_surface = category_of(first_member_name)  # 正确的地面
                    
                    # 🔥 强制选择错误的地面：从可用地面中排除正确地面
                    wrong_surfaces = [s for s in available_surfaces if s != correct_surface]
                    
                    if wrong_surfaces:
                        # 有错误地面可选，随机选一个错误的
                        assigned_surface = random.choice(wrong_surfaces)
                        print(f"  💥 {first_member_name}: {correct_surface} → {assigned_surface} (强制错误)")
                    else:
                        # 只有一种地面且恰好是正确的，跳过此物体
                        print(f"  ⚠️  {first_member_name}: 只有正确地面 {correct_surface} 可用，跳过插入")
                        skipped_groups.append(member_indices)
                        continue
                    
                    # 分配到对应地面
                    if assigned_surface == "road":
                        road_groups.append(member_indices)
                    elif assigned_surface == "sidewalk":
                        sidewalk_groups.append(member_indices)
                    elif assigned_surface == "terrain":
                        terrain_groups.append(member_indices)
            
        else:
            # 分类模式：物体按类别插入对应地面
            road_groups = []
            sidewalk_groups = []
            terrain_groups = []
            skipped_groups = []  # 记录因缺少对应地面而跳过的组
            
            for member_indices in group_members_indices:
                if not member_indices:
                    continue
                first_member_name = obj_names[member_indices[0]]
                surface_type = category_of(first_member_name)
                
                if surface_type == "sidewalk":
                    if sidewalk_pc_valid is not None:
                        sidewalk_groups.append(member_indices)
                    else:
                        # 没有 sidewalk 点云，退化到 road
                        print(f"Warning: {first_member_name} 需要 sidewalk 但不存在，退化到 road")
                        road_groups.append(member_indices)
                
                elif surface_type == "terrain":
                    if terrain_pc_valid is not None:
                        terrain_groups.append(member_indices)
                    else:
                        # 没有 terrain 点云，跳过此物体
                        print(f"Warning: {first_member_name} 需要 terrain 但不存在，跳过此物体")
                        skipped_groups.append(member_indices)
                
                else:  # "road"
                    road_groups.append(member_indices)
            
            # 如果有跳过的组，打印统计
            if skipped_groups:
                skipped_count = sum(len(g) for g in skipped_groups)
                print(f"⚠️  跳过了 {len(skipped_groups)} 个组（共 {skipped_count} 个物体），原因：缺少对应地面类型")
        
        # 预置返回数组
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
        
        # 生成位姿
        # 无论是否打乱模式，都使用分类插入（区别在于打乱模式下物体被随机分配到错误的地面）
        run_subset(road_groups, road_pc_valid, non_road_pc)
        if sidewalk_groups and (sidewalk_pc_valid is not None):
            use_non = sidewalk_non_road_pc if (sidewalk_non_road_pc is not None) else non_road_pc
            run_subset(sidewalk_groups, sidewalk_pc_valid, use_non)
        if terrain_groups and (terrain_pc_valid is not None):
            use_non = terrain_non_road_pc if (terrain_non_road_pc is not None) else non_road_pc
            run_subset(terrain_groups, terrain_pc_valid, use_non)
        
        obj_lidar_positions, obj_rz_degrees = full_obj_lidar_positions, full_obj_rz_degrees
        
        # 仅保留成功生成的物体
        success_indices = [i for i, p in enumerate(obj_lidar_positions) if p is not None]
        if not success_indices:
            print("Warning: 没有成功插入的物体")
            return False, None
        
        obj_lidar_positions = [obj_lidar_positions[i] for i in success_indices]
        obj_rz_degrees = [obj_rz_degrees[i] for i in success_indices]
        obj_names = [obj_names[i] for i in success_indices]
        obj_mesh_paths = [obj_mesh_paths[i] for i in success_indices]
        semantic_label_list = [semantic_label_list[i] for i in success_indices]
        scale_ratio_arr = [scale_ratio_arr[i] for i in success_indices]
        objs_index_arr = [i + 1 for i in range(len(success_indices))]
        
        # 生成场景信息
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
        
        # 执行数据生成（这会生成包含语义标签的点云）
        main.run(UUIDManager().get_uuid(), lidar_scene_info, modality=modality)
        
        # 读取生成后的语义标签
        # 假设语义标签保存在 dataset_output 对象中
        if hasattr(dataset_output, 'semantic_label_3d') and dataset_output.semantic_label_3d is not None:
            return True, dataset_output.semantic_label_3d
        else:
            # 如果没有保存，返回估计值
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
    """
    方法1: 分析原始场景（不插入）的语义多样性
    
    Args:
        bg_index: 背景帧索引
        
    Returns:
        包含 Gini Diversity 和统计信息的字典
    """
    container = DataGenContainer()
    dataset_config = container.dataset_config()
    dataset_type = dataset_config.get_config("demo_dataset").get("dataset_type", "")
    demo_dataset_cfg = dataset_config.get_config("demo_dataset")
    
    # 加载数据集
    dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, demo_dataset_cfg)
    
    # 加载语义标签
    semantic_labels = load_semantic_labels(dataset, bg_index)
    
    if semantic_labels is None:
        print(f"Warning: 帧 {bg_index} 没有语义标签")
        return {
            "bg_index": bg_index,
            "gini_diversity": 0.0,  # ✅ 修正：保持与 compute_semantic_statistics 一致
            "num_classes": 0,
            "total_points": 0,
            "class_distribution": {}
        }
    
    # 计算统计信息
    stats = compute_semantic_statistics(semantic_labels)
    stats["bg_index"] = bg_index
    
    print(f"\n[帧 {bg_index}] 原始场景分析:")
    print(f"  Gini Diversity: {stats['gini_diversity']:.4f}")
    print(f"  类别数: {stats['num_classes']}")
    print(f"  总点数: {stats['total_points']}")
    print(f"\n  类别分布 (Top 5):")
    sorted_classes = sorted(
        stats['class_distribution'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    for i, (class_name, info) in enumerate(sorted_classes[:5], 1):
        print(f"    {i}. {class_name}: {info['count']} points ({info['ratio']*100:.2f}%)")
    
    return stats


def generate_with_diversity_guidance(
    bg_index: int,
    category_pool: Optional[List[str]] = None,
    required_categories: Optional[List[str]] = None,
    modality: str = "pc",
    max_attempts: int = 20,
    selection_mode: str = "first_better",  # "first_better", "best_of_all", 或 "no_guidance"
    merge_surfaces: bool = False
) -> Dict:
    """
    方法2: 基于语义多样性引导插入并生成场景
    
    支持三种判断模式：
    
    模式1 - "first_better" (超过即通过，默认)：
        1. 计算原始场景的 Gini Diversity 作为基准
        2. 尝试最多 max_attempts 次插入
        3. 每次随机选择 n 个物体组进行插入
        4. 如果插入后 Gdiv > 原始 Gdiv，立即停止，保留此次结果 ✅
        5. 如果所有尝试都未超过，保留最后一次结果
    
    模式2 - "best_of_all" (取最优)：
        1. 计算原始场景的 Gini Diversity 作为基准
        2. 完整跑完 max_attempts 次插入
        3. 每次记录插入后的 Gdiv
        4. 最终选择 Gdiv 提升最大的那一次（可以是负提升）✅
        5. 如果设置了 required_categories，优先确保每个必须类别至少被尝试一次
    
    模式3 - "no_guidance" (无引导，对照实验)：
        1. 不计算或比较 Gini Diversity
        2. 随机选择 n 个物体组
        3. 尝试插入，允许最多 max_attempts 次（应对插入失败）
        4. 一旦成功插入，立即停止，不做比较 ✅
        5. 保留第一次成功的插入结果（用于对照实验）
    
    Args:
        bg_index: 背景帧索引
        category_pool: 优先选择的类别池（如 ["person", "dog", "bicycle"]）
        required_categories: 必须尝试插入的类别列表（如 ["truck", "tractor"]）
                           仅在 best_of_all 模式下生效
                           确保这些类别中的每一个都至少被尝试插入一次
        modality: 模态 ("pc", "image", "multi")
        max_attempts: 最大尝试次数（默认20次）
        selection_mode: 选择模式
            - "first_better": 超过原始值即停止（不使用 required_categories）
            - "best_of_all": 跑完所有尝试，取提升最大的（优先尝试 required_categories）
            - "no_guidance": 无引导，成功插入即停止（不使用 required_categories）
        merge_surfaces: 是否打乱物体与地面的对应关系（默认 False）
            - False: 物体按类别插入对应地面（car→road, person→sidewalk, tree→terrain）
            - True: 💥 故意打乱对应关系，随机分配物体到错误的地面
                   （例如：car可能插入到sidewalk，person可能插入到road）
                   目的：引发更多碰撞和错误场景，用于压力测试
        
    Returns:
        包含最终配置和 Gini Diversity 的字典
    """
    container = DataGenContainer()
    road_split = container.road_split()
    shapenet_loader = container.shapenet_loader()
    dataset_config = container.dataset_config()
    
    dataset_type = dataset_config.get_config("demo_dataset").get("dataset_type", "")
    dataset_output_type = dataset_config.get_config("demo_dataset").get("dataset_output_type", "")
    demo_dataset_cfg = dataset_config.get_config("demo_dataset")
    
    # 读取配置文件
    obj_insert_config_rel_path = demo_dataset_cfg.get("obj_insert_config_path", "configs/obj_insert_config.yml")
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
    
    # 构建可用的组列表
    if category_pool is not None:
        available_groups = []
        for cat in category_pool:
            matching_groups = [g for g in groups_detail.keys() if g.lower().startswith(cat.lower())]
            available_groups.extend(matching_groups)
        if not available_groups:
            print(f"Warning: 没有找到匹配 {category_pool} 的组，使用所有组")
            available_groups = list(groups_detail.keys())
    else:
        available_groups = list(groups_detail.keys())
    
    # 构建必须插入的组列表
    required_groups = []
    if required_categories is not None:
        for cat in required_categories:
            matching_groups = [g for g in groups_detail.keys() if g.lower().startswith(cat.lower())]
            required_groups.extend(matching_groups)
        if not required_groups:
            print(f"Warning: 没有找到匹配 {required_categories} 的必须类别组")
    
    print(f"\n[帧 {bg_index}] 可用组数量: {len(available_groups)}")
    if required_groups and selection_mode == "best_of_all":
        print(f"[帧 {bg_index}] 必须尝试的类别数量: {len(required_groups)} 个组")
        print(f"[帧 {bg_index}] 必须类别: {required_categories}")
        print(f"[帧 {bg_index}] 策略: 前 {len(required_groups)} 次尝试优先覆盖所有必须类别，之后随机选择")
    
    # 加载数据集
    dataset = DataFrameFactory.get_data_frame(dataset_type, bg_index, demo_dataset_cfg)
    dataset_output = DataFrameFactory.get_output_data_frame(
        dataset_type, dataset_output_type, bg_index, dataset_config.get_config("aug_dataset")
    )
    
    # ========== TODO: 修改每次插入的物体组数量 ==========
    # 修改这个值来控制每次随机选择多少个物体组进行插入
    NUM_GROUPS_PER_ATTEMPT = 1  # TODO: 每次插入 n 个物体组，可修改此值
    # ==================================================
    
    # 分析原始场景
    original_semantic_labels = load_semantic_labels(dataset, bg_index)
    if original_semantic_labels is not None:
        original_gdiv = compute_gini_diversity(original_semantic_labels)
        print(f"[原始场景] Gini Diversity: {original_gdiv:.4f}")
    else:
        original_gdiv = 0.0
        print("Warning: 无法加载原始场景的语义标签")
    
    # 根据模式初始化变量
    # 跟踪已经选过的类别前缀（所有模式都使用）
    tried_category_prefixes = set()  # 已经尝试过的类别前缀（如 "car", "truck"）
    
    if selection_mode == "first_better":
        # 模式1：超过即停止
        last_config = None  # 保存最后一次的结果
        found_better = False  # 是否找到了比原始更好的
    elif selection_mode == "best_of_all":
        # 模式2：取最优
        all_configs = []  # 保存所有尝试的结果
        best_config = None  # 保存提升最大的结果
        best_improvement = -float('inf')  # 最大提升值（可以是负数）
        # 记录已尝试的必须类别组
        tried_required_groups = set()  # 已经尝试过的必须类别组
        remaining_required_groups = required_groups.copy() if required_groups else []  # 还未尝试的必须类别组
    elif selection_mode == "no_guidance":
        # 模式3：无引导，随机插入
        pass  # 不需要初始化变量，直接进行插入
    else:
        raise ValueError(f"不支持的选择模式: {selection_mode}")
    
    # 尝试最多 max_attempts 次插入
    for attempt in range(max_attempts):
        print(f"\n--- 尝试 {attempt + 1}/{max_attempts} ---")
        
        # no_guidance 模式：完全随机，不做任何前缀去重
        if selection_mode == "no_guidance":
            # 完全随机选择，不受任何约束
            if len(available_groups) < NUM_GROUPS_PER_ATTEMPT:
                selected_groups = available_groups.copy()
            else:
                selected_groups = random.sample(available_groups, NUM_GROUPS_PER_ATTEMPT)
            
            print(f"💫 无引导随机选中 {len(selected_groups)} 个组: {selected_groups}")
        
        else:
            # first_better 和 best_of_all 模式：使用前缀去重
            # 过滤掉已经尝试过前缀的组
            untried_available_groups = [
                g for g in available_groups 
                if get_category_prefix(g) not in tried_category_prefixes
            ]
            
            if not untried_available_groups:
                print(f"⚠️  所有可用类别前缀都已尝试过，剩余尝试将从所有组中随机选择")
                untried_available_groups = available_groups
            
            print(f"未尝试的类别前缀数: {len(set([get_category_prefix(g) for g in untried_available_groups]))}")
            
            # 随机选择 n 个物体组
            if selection_mode == "best_of_all" and remaining_required_groups:
                # best_of_all 模式：优先尝试还未尝试的必须类别
                # 从剩余必须类别中选一个（也要考虑前缀去重）
                untried_required = [g for g in remaining_required_groups 
                                   if get_category_prefix(g) not in tried_category_prefixes]
                
                if untried_required:
                    selected_required = random.choice(untried_required)
                else:
                    # 如果必须类别的前缀都试过了，就从剩余必须组中随机选
                    selected_required = random.choice(remaining_required_groups)
                
                selected_groups = [selected_required]
                
                # 从必须组中移除（但前缀暂不记录，等成功后再记录）
                remaining_required_groups.remove(selected_required)
                tried_required_groups.add(selected_required)
                
                print(f"✓ 必须类别尝试: {selected_required} (前缀: {get_category_prefix(selected_required)})")
                print(f"  剩余必须类别: {len(remaining_required_groups)}")
                
                # 如果需要更多组，从untried_available_groups中补充
                remaining_count = NUM_GROUPS_PER_ATTEMPT - 1
                if remaining_count > 0:
                    # 排除已选的组和相同前缀的组
                    other_groups = [g for g in untried_available_groups if g not in selected_groups]
                    if other_groups:
                        if len(other_groups) < remaining_count:
                            selected_groups.extend(other_groups)
                        else:
                            selected_groups.extend(random.sample(other_groups, remaining_count))
            else:
                # first_better 模式：从未尝试前缀中随机选择
                if len(untried_available_groups) < NUM_GROUPS_PER_ATTEMPT:
                    selected_groups = untried_available_groups.copy()
                else:
                    selected_groups = random.sample(untried_available_groups, NUM_GROUPS_PER_ATTEMPT)
            
            print(f"随机选中 {len(selected_groups)} 个组: {selected_groups}")
        
        # 实际生成场景并获取语义标签
        success, generated_labels = _generate_scene_with_groups(
            bg_index, selected_groups, groups_detail,
            shapenet_loader, dataset, dataset_output, road_split, dataset_type, modality,
            merge_surfaces=merge_surfaces
        )
        
        if not success or generated_labels is None:
            print("场景生成失败，跳过此次尝试")
            continue
        
        # 成功插入，记录所有选中组的前缀（no_guidance 模式不记录）
        if selection_mode != "no_guidance":
            for g in selected_groups:
                tried_category_prefixes.add(get_category_prefix(g))
            print(f"✓ 成功插入！已尝试的类别前缀 ({len(tried_category_prefixes)}): {sorted(tried_category_prefixes)}")
        else:
            print(f"✓ 成功插入！（无引导模式，不记录前缀）")
        
        # 计算生成后的 Gdiv
        current_gdiv = compute_gini_diversity(generated_labels)
        current_improvement = current_gdiv - original_gdiv
        print(f"生成后 Gini Diversity: {current_gdiv:.4f} (提升: {current_improvement:+.4f})")
        
        # 保存本次配置
        current_config = {
            "groups": selected_groups,
            "gdiv": current_gdiv,
            "semantic_labels": generated_labels,
            "attempt": attempt + 1,
            "improvement": current_improvement
        }
        
        # 根据模式进行不同的处理
        if selection_mode == "first_better":
            # 模式1：超过即停止
            last_config = current_config
            
            if current_gdiv > original_gdiv:
                print(f"✅ 成功！Gdiv 超过原始值，提升: {current_improvement:.4f}")
                print(f"在第 {attempt + 1} 次尝试中找到更优解，停止搜索")
                found_better = True
                break
            else:
                print(f"❌ 未超过原始值，继续尝试...")
        
        elif selection_mode == "best_of_all":
            # 模式2：记录所有尝试，取最优
            all_configs.append(current_config)
            
            if current_improvement > best_improvement:
                best_improvement = current_improvement
                best_config = current_config
                print(f"✨ 当前最优提升: {best_improvement:+.4f}")
            else:
                print(f"   当前最优提升仍为: {best_improvement:+.4f}")
        
        elif selection_mode == "no_guidance":
            # 模式3：无引导，成功插入后立即停止
            print(f"💫 无引导模式：成功插入，立即停止（不进行 Gini 比较）")
            final_config = current_config
            break  # 成功插入即停止，不再尝试
    
    # 根据模式输出最终结果
    if selection_mode == "first_better":
        # 模式1：返回第一个超过原始值的，或最后一次的结果
        final_config = last_config
        
        if final_config is not None:
            print(f"\n{'='*60}")
            print("最终结果 [模式: 超过即通过]")
            print(f"{'='*60}")
            
            if found_better:
                print(f"状态: ✅ 找到了比原始场景更优的插入方案")
            else:
                print(f"状态: ⚠️  {max_attempts} 次尝试后未超过原始值，保留最后一次结果")
            
            print(f"最终 Gini Diversity: {final_config['gdiv']:.4f}")
            print(f"原始 Gini Diversity: {original_gdiv:.4f}")
            print(f"差值: {final_config['improvement']:+.4f}")
            print(f"选中的组: {final_config['groups']}")
            print(f"完成于第 {final_config['attempt']} 次尝试")
            
            final_stats = compute_semantic_statistics(final_config['semantic_labels'])
            print(f"类别数: {final_stats['num_classes']}")
            print(f"总点数: {final_stats['total_points']}")
            print(f"{'='*60}")
            
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
        # 模式2：返回提升最大的结果
        final_config = best_config
        
        if final_config is not None:
            print(f"\n{'='*60}")
            print("最终结果 [模式: 取最优]")
            print(f"{'='*60}")
            print(f"状态: ✨ 从 {len(all_configs)} 次成功尝试中选择了提升最大的方案")
            
            if final_config['improvement'] > 0:
                print(f"结果: ✅ 成功提升了多样性")
            elif final_config['improvement'] == 0:
                print(f"结果: ➖ 与原始值持平")
            else:
                print(f"结果: ⚠️  所有尝试都导致多样性下降，选择下降最少的")
            
            print(f"最终 Gini Diversity: {final_config['gdiv']:.4f}")
            print(f"原始 Gini Diversity: {original_gdiv:.4f}")
            print(f"差值: {final_config['improvement']:+.4f}")
            print(f"选中的组: {final_config['groups']}")
            print(f"来自第 {final_config['attempt']} 次尝试")
            
            # 显示所有尝试的提升值分布
            all_improvements = [cfg['improvement'] for cfg in all_configs]
            print(f"\n所有尝试的提升值:")
            print(f"  最大: {max(all_improvements):+.4f}")
            print(f"  最小: {min(all_improvements):+.4f}")
            print(f"  平均: {np.mean(all_improvements):+.4f} ± {np.std(all_improvements):.4f}")
            
            final_stats = compute_semantic_statistics(final_config['semantic_labels'])
            print(f"\n类别数: {final_stats['num_classes']}")
            print(f"总点数: {final_stats['total_points']}")
            print(f"{'='*60}")
            
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
        # 模式3：无引导模式，直接返回结果
        final_config = locals().get('final_config', None)
        
        if final_config is not None:
            print(f"\n{'='*60}")
            print("最终结果 [模式: 无引导]")
            print(f"{'='*60}")
            print(f"状态: 💫 随机插入，不进行多样性比较（对照实验）")
            print(f"最终 Gini Diversity: {final_config['gdiv']:.4f}")
            print(f"原始 Gini Diversity: {original_gdiv:.4f}")
            print(f"差值: {final_config['improvement']:+.4f} (仅供参考)")
            print(f"选中的组: {final_config['groups']}")
            print(f"完成于第 {final_config['attempt']} 次尝试（成功插入即停止）")
            
            final_stats = compute_semantic_statistics(final_config['semantic_labels'])
            print(f"类别数: {final_stats['num_classes']}")
            print(f"总点数: {final_stats['total_points']}")
            print(f"{'='*60}")
            
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
    
    # 如果所有尝试都失败，抛出异常让上层记录到 error_rq2.txt
    error_msg = f"所有 {max_attempts} 次尝试都失败，无法插入任何物体"
    print(f"\n{'='*60}")
    print(f"Error: {error_msg}")
    print(f"{'='*60}")
    
    # 抛出异常，让上层的 except 块捕获并记录到 error_rq2.txt
    raise RuntimeError(error_msg)


# ========== 主程序入口 ==========

if __name__ == '__main__':
    print("===== RQ2: 语义多样性引导策略 =====\n")
    
    # 读取背景帧索引
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
            # 取前100个
            indices = indices[:500]
        else:
            # 随机采样100个
            upper = MAX_BG_INDEX + 1
            k = min(200, upper)
            indices = sorted(random.sample(range(0, upper), k=k))
            with open(bgs_path, 'w', encoding='utf-8') as f:
                for v in indices:
                    f.write(f"{v}\n")
    except Exception as e:
        print(f"Failed preparing bg indices: {e}")
        indices = [0]
    
    print(f"将处理 {len(indices)} 帧\n")
    
    # ========== 第一步：分析所有原始场景的多样性 ==========
    print("=" * 60)
    print("第一步：分析原始场景（不插入）")
    print("=" * 60)
    
    original_results = []
    for ix, bg_index in enumerate(indices, start=1):
        try:
            print(f"\n处理帧 {ix}/{len(indices)}: BG_INDEX={bg_index}")
            stats = analyze_original_diversity(bg_index)
            original_results.append(stats)
        except Exception as e:
            print(f"Error at frame {bg_index}: {e}")
            continue
    
    # 保存原始场景分析结果
    with open(ORIGINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(original_results, f, indent=2)
    
    # 统计汇总
    if original_results:
        gdiv_values = [r["gini_diversity"] for r in original_results if r["gini_diversity"] > 0]
        if gdiv_values:
            print(f"\n{'='*60}")
            print("原始场景多样性统计:")
            print(f"  平均 Gini Diversity: {np.mean(gdiv_values):.4f} ± {np.std(gdiv_values):.4f}")
            print(f"  最小值: {np.min(gdiv_values):.4f}")
            print(f"  最大值: {np.max(gdiv_values):.4f}")
            print(f"  结果已保存到: {ORIGINAL_OUTPUT_FILE}")
            print(f"{'='*60}\n")
    
    # ========== 第二步：使用多样性引导插入生成 ==========
    print("\n" + "=" * 60)
    print("第二步：基于多样性引导插入生成")
    print("=" * 60)
    
    # ========== TODO: 选择判断模式 ==========
    # 模式1 "first_better": 超过原始 Gdiv 即停止（快速，适合大多数情况）
    # 模式2 "best_of_all": 跑完所有尝试，取提升最大的（全面，适合对比实验）
    # 模式3 "no_guidance": 无引导，随机插入一次，不做比较（对照实验）
    SELECTION_MODE = "best_of_all"  # TODO: 修改这里切换模式
    # SELECTION_MODE = "no_guidance"
    # ==========================================
    
    # ========== TODO: 地面打乱选项（压力测试用） ==========
    # False: 物体按类别插入对应地面（car→road, person→sidewalk, tree→terrain）
    # True:  💥 故意打乱对应关系，随机分配物体到错误的地面
    #        例如：car可能插入到sidewalk，person可能插入到road
    #        目的：引发更多碰撞和错误场景，用于压力测试
    MERGE_SURFACES = True # TODO: 修改为 True 启用地面打乱模式
    # =======================================================
    
    # 设置稀有类别池（优先选择这些稀有类别）
    category_pool = ["car", "bus", "truck", "motorcycle", "bicycle", "tractor", "ambulance",
                     "tree", "adult_1", "traffic_sign_1", "fence_2",
                     "telephone_pole_1",
                     "person-pulling-luggage", "person-pushing-stroller", "stroller", "wheelchair",
                     "dog_1",  "cat_1", "boar_1", "boar_2",
                     "Shrubbery", "person-riding-bicycle_1", "person-riding-bicycle_2", "person-riding-motor", "person-pushing-bicycle",
                     "person-pushing-motorcycle", "dog-walker_1"]

    # category_pool = []
    required_categories = ["tractor_1", "truck_1"]
    # required_categories = []
    
    print(f"判断模式: {SELECTION_MODE}")
    print(f"地面合并: {MERGE_SURFACES}")
    print(f"优先类别池: {category_pool}")
    print(f"必须类别: {required_categories}\n")
    
    generated_results = []
    for ix, bg_index in enumerate(indices, start=1):
        try:
            print(f"\n{'='*60}")
            print(f"处理帧 {ix}/{len(indices)}: BG_INDEX={bg_index}")
            print(f"{'='*60}")
            
            result = generate_with_diversity_guidance(
                bg_index=bg_index,
                category_pool=category_pool,
                required_categories=required_categories,
                modality="pc",
                max_attempts=5,
                selection_mode=SELECTION_MODE,  # 使用配置的模式
                merge_surfaces=MERGE_SURFACES   # 使用配置的地面合并选项
            )
            
            generated_results.append(result)
            
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"❌ Error at frame {bg_index}: {e}")
            print(f"{'!'*60}")
            import traceback
            traceback.print_exc()
            print(f"{'!'*60}\n")
            
            # 记录错误帧到 error_rq2.txt
            try:
                with open(error_path, 'a', encoding='utf-8') as ef:
                    ef.write(f"{bg_index}\n")
                print(f"✓ 错误帧已记录到: {error_path}")
            except Exception as write_error:
                print(f"Warning: 无法写入错误文件: {write_error}")
            
            continue
    
    # 保存生成结果
    with open(GENERATED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(generated_results, f, indent=2)
    
    # 最终统计汇总
    if generated_results:
        success_results = [r for r in generated_results if r.get("success", True)]
        if success_results:
            improvements = [r["improvement"] for r in success_results]
            final_gdivs = [r["final_gdiv"] for r in success_results]
            
            print(f"\n{'='*60}")
            print("最终统计汇总:")
            print(f"{'='*60}")
            print(f"总帧数: {len(indices)}")
            print(f"成功帧数: {len(success_results)}")
            print(f"失败帧数: {len(indices) - len(success_results)}")
            print(f"\n平均 Gini Diversity 提升: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
            print(f"最小提升: {np.min(improvements):.4f}")
            print(f"最大提升: {np.max(improvements):.4f}")
            print(f"\n生成后平均 Gini Diversity: {np.mean(final_gdivs):.4f} ± {np.std(final_gdivs):.4f}")
            print(f"\n结果已保存到:")
            print(f"  原始场景: {ORIGINAL_OUTPUT_FILE}")
            print(f"  生成场景: {GENERATED_OUTPUT_FILE}")
            if error_path.exists():
                print(f"  错误记录: {error_path}")
            print(f"{'='*60}\n")
    
    print("RQ2 完成！")

