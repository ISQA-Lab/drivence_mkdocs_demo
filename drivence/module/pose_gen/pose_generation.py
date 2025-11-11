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

    def __init__(self):
        self.max_try_pose_num = 50
        self.collision_detector = CollisionDetector()
        self.max_non_road_points_limit = 300

    def generate_pose(self, init_mesh_obj: o3d.geometry.TriangleMesh, road_pc_input: numpy.ndarray,
                      non_road_pc: numpy.ndarray, init_objs_box3d_corners: numpy.ndarray,
                      objs_box3d_corners: numpy.ndarray) -> Tuple[List[float], float]:
        """
        Generate a collision-free pose for a single mesh.

        Args:
            init_mesh_obj: Candidate mesh to be inserted.
            road_pc_input: Candidate ground points.
            non_road_pc: Non-ground points for exclusion checks.
            init_objs_box3d_corners: Bounding boxes of the original background objects.
            objs_box3d_corners: Bounding boxes of the already inserted objects.

        Returns:
            (position, yaw_degree) if a valid pose is found; otherwise None.
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
        Internal pose search that samples from road points.

        Args:
            mesh_obj: Candidate mesh.
            road_pc_input: Road surface points.

        Returns:
            (position, yaw_degree) when successful, otherwise (None, None).
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
        Apply translation and yaw rotation to a mesh.

        Args:
            mesh_obj: Mesh to transform.
            shift: Translation vector.
            rotation: Rotation around the Z axis in degrees.

        Returns:
            Transformed mesh.
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
        Uniformly scale a mesh around its center.

        Args:
            mesh_obj: Mesh to scale.
            scale_ratio: Scaling factor.

        Returns:
            Scaled mesh.
        """
        if scale_ratio == 1:
            pass
        else:
            mesh_obj.scale(scale=scale_ratio, center=mesh_obj.get_center())
        return mesh_obj

    def _is_on_road(self, mesh_obj: o3d.geometry.TriangleMesh, non_road_pc: numpy.ndarray) -> bool:
        """
        Detect whether the mesh overlaps non-road points.

        Args:
            mesh_obj: Mesh to test.
            non_road_pc: Points that should remain obstacle-free.

        Returns:
            True if the mesh stays on the road surface, False otherwise.
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
    Generate poses for groups of meshes with optional relative offsets.

    Args:
        road_pc_valid: Road surface points available for placement.
        non_road_pc: Points that represent obstacles.
        calib_info: Calibration records for the background scene.
        dataset: Dataset frame used for collision queries.
        mesh_list: List of meshes to position.
        group_members_indices: Indices describing each insertion group.
        relative_displacement_list: Per-mesh displacement relative to the group base.
        relative_rotation_list: Optional per-mesh rotation offset (radians).

    Returns:
        Tuple of (positions, yaw_degrees) aligned with the order of mesh_list.
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
