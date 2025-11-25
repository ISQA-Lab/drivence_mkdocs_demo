from typing import Tuple, List

import numpy as np
import copy
import math
import open3d as o3d
import numpy
from drivence.module.pose_gen.collision_detection import CollisionDetector
from drivence.utils import format_convert, box_utils
from drivence.utils.common_utils import extract_initial_objs_from_bg
import os


class PoseGenerator(object):
    """
    姿态生成器类，用于为待插入场景的网格模型生成无碰撞、贴合路面的3D姿态（位置+偏航角）。

    核心功能：
    1. 基于路面点云采样候选位置，确保模型贴合地面；
    2. 进行路面有效性校验（避免模型落在非路面区域）；
    3. 进行碰撞检测（避免与背景物体、已插入物体重叠）；
    4. 支持最大尝试次数限制，确保流程高效终止。
    """
    def __init__(self):
        """
        初始化姿态生成器，配置核心参数和依赖组件。
        """
        self.max_try_pose_num = 50
        self.collision_detector = CollisionDetector()
        self.max_non_road_points_limit = 300

    def generate_pose(self, init_mesh_obj: o3d.geometry.TriangleMesh, road_pc_input: numpy.ndarray,
                      non_road_pc: numpy.ndarray, init_objs_box3d_corners: numpy.ndarray,
                      objs_box3d_corners: numpy.ndarray) -> Tuple[List[float], float]:
        """
        为单个网格模型生成无碰撞的3D姿态（位置+绕Z轴偏航角）。

        核心流程：
        1. 循环尝试生成姿态（最多max_try_pose_num次）；
        2. 采样路面点作为候选位置，生成偏航角，变换网格模型；
        3. 校验模型是否在路面上（非路面点数不超过阈值）；
        4. 校验模型是否与背景物体、已插入物体碰撞；
        5. 找到满足条件的姿态则返回，全部尝试失败则返回None。

        Args:
            init_mesh_obj (o3d.geometry.TriangleMesh): 待插入的候选网格模型（原始未变换状态）；
            road_pc_input (numpy.ndarray): 路面点云（用于采样候选位置，确保模型贴合地面），形状为 (N, 3+)；
            non_road_pc (numpy.ndarray): 非路面点云（用于校验模型是否落在路面上），形状为 (M, 3+)；
            init_objs_box3d_corners (numpy.ndarray): 原始背景物体的3D包围盒角点数组，形状为 (K, 8, 3)，
                K为背景物体数量，每个物体对应8个3D角点；
            objs_box3d_corners (numpy.ndarray): 已插入物体的3D包围盒角点数组，格式与 init_objs_box3d_corners 一致。

        Returns:
            Tuple[List[float], float]: 有效姿态（位置+偏航角），格式为 ([x, y, z], yaw_degree)；
                若所有尝试均失败，返回 None。
        """
        max_try_pose_num = self.max_try_pose_num
        while max_try_pose_num > 0:
            mesh_obj = copy.deepcopy(init_mesh_obj)
            position, rz_degree = self._generate_pose_detail(mesh_obj, road_pc_input)
            mesh_obj = PoseGenerator.transform_mesh_by_pose(mesh_obj, position, rz_degree)
            obj_inserted_box3d = mesh_obj.get_minimal_oriented_bounding_box()
            obj_box3d_adjusted = box_utils.covert_boxo3d_to_aligned_boxo3d(obj_inserted_box3d)
            obj_box3d_corners = box_utils.box_o3d_to_corners3d(obj_box3d_adjusted)

            is_on_road_flag = self._is_on_road(mesh_obj, non_road_pc)
            if max_try_pose_num % 10 == 0:
                print(f"There are {max_try_pose_num} chances left,is_on_road_flag:", is_on_road_flag)
                print(f"position:{position},rz_degree:{rz_degree}")
            if not is_on_road_flag:
                max_try_pose_num -= 1
                continue

            is_collision_flag = self.collision_detector.collision_detection(init_objs_box3d_corners,
                                                                            objs_box3d_corners,
                                                                            obj_box3d_corners)

            if is_collision_flag:
                max_try_pose_num -= 1
                print("Collision check failed")
                continue
            else:
                return position, rz_degree

        return None

    def _generate_pose_detail(self, mesh_obj: o3d.geometry.TriangleMesh, road_pc_input: numpy.ndarray) -> Tuple[
        List[float], float]:
        """
        内部姿态采样方法：基于路面点云随机采样候选位置，结合网格尺寸调整Z轴高度，随机生成偏航角。

        核心逻辑：
        1. 限制最大采样次数（40次），避免无限循环；
        2. 过滤路面点云（仅保留X>5的区域，可通过注释代码扩展过滤规则）；
        3. 随机采样路面点，校验采样区域的点云密度（避免稀疏区域）；
        4. 根据网格高度调整Z轴位置（确保模型底部贴合路面）；
        5. 随机生成偏航角（根据Y坐标正负调整方向）。

        Args:
            mesh_obj (o3d.geometry.TriangleMesh): 待插入的候选网格模型（用于获取尺寸信息）；
            road_pc_input (numpy.ndarray): 路面点云（用于采样候选位置），形状为 (N, 3+)。

        Returns:
            Tuple[List[float], float]: 候选姿态（[x, y, z], yaw_degree）；
                若采样失败（如无可用路面点、密度校验多次失败），返回 (None, None)。
        """
        cnt = 0
        min_x = 7
        while (True):
            if cnt > 40:
                return None, None
            cnt += 1

            min_xyz = mesh_obj.get_min_bound()
            max_xyz = mesh_obj.get_max_bound()
            half_height = (max_xyz[2] - min_xyz[2]) / 2

            road_pc = road_pc_input.copy()

            # Optional filter hook for custom placement rules (disabled by default)
            # road_pc = road_pc[road_pc[:, 0] < 15]
            # road_pc = road_pc[road_pc[:, 1] < 1]
            # road_pc = road_pc[road_pc[:, 1] > -1]
            road_pc = road_pc[road_pc[:, 0] > 5]

            if len(road_pc) == 0:
                road_pc = road_pc_input.copy()

            if len(road_pc) == 0:
                print("[warn] No available road points for pose sampling")
                return None

            # Random sampling from the candidate road points
            sample_index = np.random.randint(0, len(road_pc))

            # Local density check to avoid sparse regions
            selected_point = road_pc[sample_index][:3]
            x, y, z = selected_point
            # Distance to every other point
            distances = np.sqrt(np.sum((road_pc[:, :3] - selected_point) ** 2, axis=1))
            # Count neighbors within a fixed radius
            radius = 2.0
            min_neighbors = 10
            neighbors_count = np.sum(distances < radius)

            # If not enough neighbors, resample
            if neighbors_count < min_neighbors:
                print(f"Density check failed: Selected point ({x:.2f}, {y:.2f}, {z:.2f}) has only {neighbors_count} points within {radius}m radius, less than {min_neighbors}. Resampling.")
                continue

            print(f"Density check passed: Selected point ({x:.2f}, {y:.2f}, {z:.2f}) has {neighbors_count} points within {radius}m radius")

            if x < min_x:
                if cnt > 20: min_x = 3
                elif cnt > 10: min_x = 5
                continue

            position = [x, y, z + half_height]
            position = [round(i, 2) for i in position]
            rz_degree = np.random.randint(2, 10)

            if y > 0:
                rz_degree = -rz_degree

            break
        
        return position, rz_degree

    @staticmethod
    def transform_mesh_by_pose(mesh_obj: o3d.geometry.TriangleMesh, shift: List[float] = None,
                               rotation: float = None) -> o3d.geometry.TriangleMesh:
        """
        静态方法：根据姿态（平移+偏航角）变换网格模型。

        变换顺序：先平移（基于shift参数），后绕Z轴旋转（基于rotation参数，单位为度）。

        Args:
            mesh_obj (o3d.geometry.TriangleMesh): 待变换的网格模型；
            shift (List[float], 可选): 平移向量，格式为 [x, y, z]，默认None（不平移）；
            rotation (float, 可选): 绕Z轴的偏航角（单位：度），默认None（不旋转）。

        Returns:
            o3d.geometry.TriangleMesh: 变换后的网格模型（原地变换，返回原对象）。
        """
        if shift is not None:
            mesh_obj.translate(shift)
        if rotation is not None:
            rz_radians = math.radians(rotation)
            RZ = mesh_obj.get_rotation_matrix_from_xyz((0, 0, -rz_radians))
            mesh_obj.rotate(RZ)
        return mesh_obj

    @staticmethod
    def scale_mesh(mesh_obj: o3d.geometry.TriangleMesh, scale_ratio: float) -> o3d.geometry.TriangleMesh:
        """
        静态方法：对网格模型进行均匀缩放（围绕网格中心点）。

        Args:
            mesh_obj (o3d.geometry.TriangleMesh): 待缩放的网格模型；
            scale_ratio (float): 缩放系数（1.0表示不缩放，>1放大，<1缩小）。

        Returns:
            o3d.geometry.TriangleMesh: 缩放后的网格模型（原地缩放，返回原对象）。
        """
        if scale_ratio == 1:
            pass
        else:
            mesh_obj.scale(scale=scale_ratio, center=mesh_obj.get_center())
        return mesh_obj

    def _is_on_road(self, mesh_obj: o3d.geometry.TriangleMesh, non_road_pc: numpy.ndarray) -> bool:
        """
        校验网格模型是否落在路面上：通过统计模型包围盒内的非路面点数判断。

        核心逻辑：若模型包围盒内包含的非路面点数≥阈值，则判定为未在路面上；否则判定为在路面上。

        Args:
            mesh_obj (o3d.geometry.TriangleMesh): 待校验的网格模型（已变换到候选姿态）；
            non_road_pc (numpy.ndarray): 非路面点云，形状为 (M, 3+)。

        Returns:
            bool: 模型是否在路面上的结果：
                - True：包围盒内非路面点数 < 阈值（在路面上）；
                - False：包围盒内非路面点数 ≥ 阈值（未在路面上）。
        """
        box = mesh_obj.get_oriented_bounding_box()
        non_road_pcd = format_convert.pc_numpy_2_pcd(non_road_pc)
        non_road_pcd_contained = non_road_pcd.crop(box)

        if len(non_road_pcd_contained.points) >= self.max_non_road_points_limit:
            return False
        else:
            return True


def generate_group_pose_from_road(road_pc_valid, non_road_pc, calib_info, dataset, mesh_list, group_members_indices, relative_displacement_list, relative_rotation_list=None):
    """
    为分组的网格模型生成姿态（位置+偏航角），支持组内模型基于基准模型的相对偏移（位移+旋转），
    确保组内模型无碰撞、贴合路面，且不与背景物体/已插入物体重叠。

    核心逻辑：
    1. 为每个分组选择一个基准模型，生成其无碰撞的基础姿态；
    2. 组内其他模型基于基准姿态和预设的相对偏移（位移+旋转）计算自身姿态；
    3. 校验组内所有模型与背景物体的碰撞情况，整体通过则保留组内所有姿态；
    4. 收集所有模型的最终姿态，返回与输入mesh_list顺序一致的结果。

    Args:
        road_pc_valid (np.ndarray): 有效路面点云（用于基准模型姿态采样），形状为 (N, 3+)；
        non_road_pc (np.ndarray): 非路面点云（用于校验模型是否在路面上），形状为 (M, 3+)；
        calib_info (dict): 背景场景的标定信息（用于提取背景物体包围盒）；
        dataset: 数据集帧对象（需支持 get_label() 方法，若有 get_label_2() 方法则用于生成目标检测标注）；
        mesh_list (List[o3d.geometry.TriangleMesh]): 待生成姿态的所有网格模型列表，
            顺序与最终返回的姿态列表一致；
        group_members_indices (List[List[int]]): 分组索引列表，每个元素为一个子列表，
            子列表包含对应分组内模型在 mesh_list 中的索引（如 [[0,1], [2,3,4]] 表示2个分组）；
        relative_displacement_list (List[List[float]]): 每个模型相对于其分组基准模型的位移偏移列表，
            长度与 mesh_list 一致，每个元素为 [dx, dy, dz]（单位：米）；
        relative_rotation_list (Optional[List[float]], 可选): 每个模型相对于其分组基准模型的旋转偏移列表，
            长度与 mesh_list 一致，每个元素为绕Z轴的旋转角（单位：弧度），默认 None（无旋转偏移）。

    Returns:
        Tuple[List[Optional[List[float]]], List[Optional[float]]]: 所有模型的姿态结果，与 mesh_list 顺序一致：
            - 第一个元素：位置列表，每个元素为 [x, y, z]（保留2位小数），失败则为 None；
            - 第二个元素：偏航角列表（单位：度），失败则为 None。
    """
    pose_generator = PoseGenerator()
    obj_lidar_positions = [None] * len(mesh_list)
    obj_rz_degrees = [None] * len(mesh_list)

    # Background objects for collision detection
    label = dataset.get_label()
    label_2_path = None
    if hasattr(dataset, 'get_label_2'):
        label_2_path = dataset.get_label_2()
    bg_objs_bounding_box = extract_initial_objs_from_bg(calib_info, label, label_2_path=label_2_path)
    inserted_box_corners = []  # Bounding boxes of already inserted objects

    for member_indices in group_members_indices:
        if not member_indices:
            continue

        # Search for a base member; fall back to other members when necessary
        base_found = False
        base_idx = None
        base_position = None
        base_rz_degree = None
        for candidate_idx in member_indices:
            candidate_mesh = mesh_list[candidate_idx]
            pose_params = pose_generator.generate_pose(
                candidate_mesh,
                road_pc_valid,
                non_road_pc,
                bg_objs_bounding_box,
                np.asarray(inserted_box_corners)
            )
            if pose_params is None:
                continue
            base_position, base_rz_degree = pose_params
            base_idx = candidate_idx
            try:
                mesh_tmp = copy.deepcopy(candidate_mesh)
                mesh_tmp = PoseGenerator.transform_mesh_by_pose(mesh_tmp, base_position, base_rz_degree)
                obj_inserted_box3d = mesh_tmp.get_minimal_oriented_bounding_box()
                obj_box3d_adjusted = box_utils.covert_boxo3d_to_aligned_boxo3d(obj_inserted_box3d)
                obj_box3d_corners = box_utils.box_o3d_to_corners3d(obj_box3d_adjusted)
                inserted_box_corners.append(obj_box3d_corners)
                obj_lidar_positions[base_idx] = [round(v, 2) for v in base_position]
                obj_rz_degrees[base_idx] = base_rz_degree
                base_found = True
            except Exception as e:
                print(f"Failed to update inserted object bounding box: {e}")
                base_found = False
                base_idx = None
                base_position = None
                base_rz_degree = None
                continue
            if base_found:
                break

        if not base_found:
            print(f"Pose generation failed for group {member_indices}, skipping group")
            continue

        # Apply relative pose to the remaining members and validate against background objects
        group_positions = []
        group_rotations = []
        group_box_corners = []
        group_valid = True
        
        for idx in member_indices:
            relative_displacement = relative_displacement_list[idx]
            adjusted_position = [
                base_position[0] + relative_displacement[0],
                base_position[1] + relative_displacement[1],
                base_position[2] + relative_displacement[2]
            ]

            if relative_rotation_list is not None and idx < len(relative_rotation_list):
                relative_rotation_rad = relative_rotation_list[idx]
                adjusted_rz_degree = base_rz_degree + math.degrees(relative_rotation_rad)
            else:
                adjusted_rz_degree = base_rz_degree

            try:
                mesh_tmp = copy.deepcopy(mesh_list[idx])
                mesh_tmp = PoseGenerator.transform_mesh_by_pose(mesh_tmp, adjusted_position, adjusted_rz_degree)
                obj_inserted_box3d = mesh_tmp.get_minimal_oriented_bounding_box()
                obj_box3d_adjusted = box_utils.covert_boxo3d_to_aligned_boxo3d(obj_inserted_box3d)
                obj_box3d_corners = box_utils.box_o3d_to_corners3d(obj_box3d_adjusted)

                is_collision_flag = pose_generator.collision_detector.collision_detection(
                    bg_objs_bounding_box,
                    np.asarray([]),
                    obj_box3d_corners
                )

                if is_collision_flag:
                    print(f"Collision detected between member {idx} and background objects. Skipping entire group.")
                    group_valid = False
                    break

                # Collect outcomes for the current group
                group_positions.append((idx, adjusted_position))
                group_rotations.append((idx, adjusted_rz_degree))
                group_box_corners.append(obj_box3d_corners)
                
            except Exception as e:
                print(f"Failed to compute pose for member {idx}: {e}")
                group_valid = False
                break

        if group_valid:
            for idx, position in group_positions:
                obj_lidar_positions[idx] = [round(v, 2) for v in position]
            for idx, rotation in group_rotations:
                obj_rz_degrees[idx] = rotation
            inserted_box_corners.extend(group_box_corners)
            print(f"Group {member_indices} inserted successfully with {len(member_indices)} members")
        else:
            print(f"Group {member_indices} failed collision checks and was skipped")

    return obj_lidar_positions, obj_rz_degrees
