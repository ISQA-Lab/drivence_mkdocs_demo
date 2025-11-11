import copy
import numpy as np
import os
import shapely
from natsort import natsorted
from shapely.geometry import box, Polygon
from shapely import affinity
import open3d as o3d

from drivence.utils.format_convert import to_torch
from drivence.utils.pc_utils import rotate_points_along_z
from drivence.utils.transform_utils import get_euler_from_rotate_matrix


# corner 4*2
def sort_corners_clockwise(corner):
    corner = np.array(corner)

    # Compute centroid
    centroid = np.mean(corner, axis=0)

    # Calculate polar angle relative to the centroid
    angles = np.arctan2(corner[:, 1] - centroid[1], corner[:, 0] - centroid[0])

    # Sort by angle in ascending order (clockwise)
    sorted_corners = corner[np.argsort(angles)]

    return sorted_corners


# [x, y, z, dx, dy, dz, heading]
# Note: coordinate axes must follow the KITTI LiDAR convention
def lidar_boxesn7_to_corners3d(boxes3d: np.ndarray, return_type: str = "numpy") -> np.ndarray:
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns: (N, 8, 3) array of box corners in the LiDAR coordinate system
    """
    boxes3d = to_torch(boxes3d)
    pad_flag = False
    if len(boxes3d.shape) == 1:
        pad_flag = True
        boxes3d = boxes3d[np.newaxis, :]

    assert boxes3d.shape[1] == 7

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6], return_type="torch").view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    result = corners3d
    if return_type == "numpy":
        result = corners3d.numpy()
    if pad_flag:
        result = result[0]
    return result


def corners3d_to_boxesn7(corners3d: np.ndarray, return_type: str = "numpy") -> np.ndarray:
    ...


def box2d_overlapping_area(box2d_a, box2d_b):
    x_min_a, y_min_a, x_max_a, y_max_a = box2d_a
    x_min_b, y_min_b, x_max_b, y_max_b = box2d_b
    dx = min(x_max_a, x_max_b) - max(x_min_a, x_min_b)
    dy = min(y_max_a, y_max_b) - max(y_min_a, y_min_b)
    if (dx > 0) and (dy > 0):
        return dx * dy
    else:
        return 0


# The eight corners follow Open3D's ordering
def box_o3d_to_corners3d(boxo3d: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    """Return the eight corners of an Open3D oriented bounding box."""
    corner_box = np.asarray(boxo3d.get_box_points())
    return corner_box


def corners3d_to_box_o3d(corners3d: np.ndarray) -> o3d.geometry.OrientedBoundingBox:
    box3d = o3d.geometry.OrientedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(corners3d))
    return box3d

def covert_boxo3d_to_aligned_boxo3d(boxo3d: o3d.geometry.OrientedBoundingBox) -> o3d.geometry.OrientedBoundingBox:
    """Align an oriented bounding box with the ground plane using Euler angles."""
    # sciangle_0, sciangle_1, sciangle_2 = UtilsBox.get_euler_from_matrix(UtilsBox.get_box3d_R(box3d))
    sciangle_0, sciangle_1, sciangle_2 = get_euler_from_rotate_matrix(copy.copy(boxo3d.R))

    if 2.5 > abs(sciangle_0) > 1.47:
        ...

    if abs(sciangle_0) > 2.5 or abs(sciangle_1) > 3 or abs(sciangle_2) > 3:
        R_return = boxo3d.get_rotation_matrix_from_zyx([-sciangle_2, -sciangle_1, - sciangle_0])
        boxo3d.rotate(R_return)
        R_return_1 = boxo3d.get_rotation_matrix_from_xyz([0, 0, -sciangle_2])
        boxo3d.rotate(R_return_1)
    else:
        R_return = boxo3d.get_rotation_matrix_from_zyx([-sciangle_2, -sciangle_1, - sciangle_0])
        boxo3d.rotate(R_return)
        R_return_1 = boxo3d.get_rotation_matrix_from_xyz([0, 0, sciangle_2])
        boxo3d.rotate(R_return_1)

    return boxo3d


from drivence.dataset.common.label import Label


def get_bounding_box(bounding_box_dir: str) -> Label:
    # bounding_box_dir = blender_outputs_dict["bounding_box"]
    bounding_box_files = natsorted(os.listdir(bounding_box_dir))
    data = []
    for bounding_box_file in bounding_box_files:
        obj_name = bounding_box_file.split(".")[0].split("_")[1] # {obj_idx}_{obj_name}
        bounding_box_filepath = os.path.join(bounding_box_dir, bounding_box_file)
        corners = []
        with open(bounding_box_filepath, 'r') as file:
            lines = file.readlines()
            lines = [line.strip().split(" ") for line in lines]
            for line in lines:
                corners.append([float(line[0]), float(line[1]), float(line[2])])
        # corners = np.array(lines)
        data.append({"obj_name": obj_name, "corners": np.array(corners)})
    return data


class Box3D(object):
    def __init__(self, box3d_corners=None):
        self.box3d_corners = box3d_corners
        # self.calibration = None

    # def set_calibration(self,calibration):
    #     self.calibration = calibration

    def from_mesh(self, mesh_obj, align_to_axis=False):
        # get_minimal_oriented_bounding_box may yield a box that is not aligned with the z-axis
        # Open3D's align_bounding_box enforces alignment with the xy-plane, so apply manual corrections here
        # When align_to_axis=True the result is axis-aligned with z and orthogonal to the ground plane
        boxo3d = mesh_obj.get_minimal_oriented_bounding_box()
        if align_to_axis:
            boxo3d = covert_boxo3d_to_aligned_boxo3d(boxo3d)
        self.box3d_corners = box_o3d_to_corners3d(boxo3d)

    def from_box3d(self, box3d_corners):
        self.box3d_corners = box3d_corners

    def from_boxo3d(self, boxo3d: o3d.geometry.OrientedBoundingBox):
        self.box3d_corners = box_o3d_to_corners3d(boxo3d)

    def get_o3d_box(self):
        boxo3d = corners3d_to_box_o3d(self.box3d_corners)
        return boxo3d

    def get_box_range(self, return_points=False):
        x_min, y_min, z_min = np.min(self.box3d_corners, axis=0)
        x_max, y_max, z_max = np.max(self.box3d_corners, axis=0)
        if return_points:
            return (x_min, y_min, z_min), (x_max, y_max, z_max)
        else:
            return x_min, x_max, y_min, y_max, z_min, z_max

    # Keep the top-four corners (bird's-eye view)
    def to_box_bev(self):
        box3d_corners = self.box3d_corners
        sorted_indices = np.argsort(box3d_corners[:, 2])[::-1]
        top_indices = sorted_indices[:4]
        top_corners = box3d_corners[top_indices][:, :2]
        top_corners = sort_corners_clockwise(top_corners)
        return top_corners

    def get_box_bottom_center(self):
        box_range = self.get_box_range()
        x_min, x_max, y_min, y_max, z_min, z_max = box_range
        bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
        return bottom_center

    def get_box_size(self):
        o3d_box = self.get_o3d_box()
        x_, y_, z_ = np.asarray(o3d_box.extent)
        h, w, l = z_, y_, x_
        return h, w, l

    def get_box_center(self):
        box_range = self.get_box_range(return_points=True)
        center = np.mean(box_range, axis=0)
        return center


class LidarBox3D(Box3D):
    def __init__(self, box3d_corners=None):
        super().__init__(box3d_corners)
        # self.calibration = None

    def from_lidar_box3d_n7(self, box3d_n7):  # only for lidar box3d
        box3d_corners = lidar_boxesn7_to_corners3d(box3d_n7)
        self.box3d_corners = box3d_corners

    def to_box3d_n7(self):  # x,y,z,h,w,l,ry
        ...


# when yaw=0 align to positive y-axis     
#    (y-axis)
#      |/ yaw = 60°  yaw is measured clockwise from the positive x-axis 
# -------------  # horizontal axis (x-axis)
#      |  
#      |  
# box2d_center (x_center,y_center,w (dx),l (dy))  
# box_diagonal(x_min,y_min,x_max,y_max), (left,top,right,bottom), x is width, y is length
# box2d_corners (N,4,2)

def box2d_diagonal_to_center(left, top, right, bottom):
    # Compute center, width, and length
    x_center = (left + right) / 2
    y_center = (top + bottom) / 2
    width = right - left
    length = top - bottom  # Length along the y-axis
    assert width > 0 and length > 0
    return x_center, y_center, width, length


def box2d_center_to_diagonal(x_center, y_center, width, length):
    # Compute left, top, right, and bottom
    half_width = width / 2
    half_length = length / 2
    left = x_center - half_width
    top = y_center - half_length  # y_max
    right = x_center + half_width
    bottom = y_center + half_length  # y_min
    return left, top, right, bottom



def box2d_corner_to_diagonal(box2d_corners):
    x_center = np.mean(box2d_corners[:, 0])
    y_center = np.mean(box2d_corners[:, 1])
    width = np.max(box2d_corners[:, 0]) - np.min(box2d_corners[:, 0])
    length = np.max(box2d_corners[:, 1]) - np.min(box2d_corners[:, 1])
    assert 1 == 2
    return x_center, y_center, width, length


class Box2D(object):
    def __init__(self):
        self.shapely_box = None
        self.box2d_diagonal = None  # (x_min,y_min,x_max,y_max) box2d_diagonal is the aligned box2d

    # Handles either rotated or axis-aligned corners
    @classmethod
    def from_corners(cls, corners):
        box2d = cls()
        # box2d.shapely_box = box
        box2d.shapely_box = Polygon(corners)
        box2d.box2d_diagonal = box2d.shapely_box.bounds  # (x_min,y_min,x_max,y_max)
        return box2d

    # box2d_center before rotation and the yaw to apply
    @classmethod
    def from_box2d_center(cls, box2d_center, yaw=0):  # # box2d (x_center,y_center,w (dx),l (dy),yaw)
        box2d = cls()
        box2d_diagonal = box2d_center_to_diagonal(*box2d_center)  # ???
        # print("box2d_diagonal",box2d_diagonal)
        shapely_box = shapely.box(*box2d_diagonal)
        box2d.shapely_box = shapely_box
        box2d.box2d_diagonal = shapely_box.bounds
        box2d.rotate(yaw)
        return box2d

    # box2d diagonal before rotation and the corresponding yaw
    @classmethod
    def from_box2d_diagonal(cls, box2d_diagonal,
                            yaw=0):  # (N,4) (x_min,y_min,x_max,y_max) box2d_diagonal is the aligned box2d
        # box_center = box2d_diagonal_to_center(box2d_diagonal)
        shapely_box = box(*box2d_diagonal)
        box2d = cls()
        box2d.shapely_box = shapely_box
        box2d.box2d_diagonal = shapely_box.bounds
        box2d.rotate(yaw)
        return box2d

    def show(self, color="red", alpha=0.5, axis_equal=False):
        from matplotlib import pyplot as plt
        x, y = self.get_exterior()
        plt.fill(x, y, color=color, alpha=alpha)
        if axis_equal:
            plt.axis("equal")

    def is_intersect(self, other_box2d):
        return self.shapely_box.intersects(other_box2d.shapely_box)

    def rotate(self, yaw=None):
        # Positive angles represent counter-clockwise rotation; negative angles are clockwise
        self.shapely_box = affinity.rotate(self.shapely_box, yaw, origin="centroid", use_radians=True)

    def get_exterior(self):
        return np.array(self.shapely_box.exterior.xy)

    def get_corners(self):
        return self.get_exterior().T[:-1]

    def truncate(self, image_shape, return_ratio=True):
        # TODO: investigate required adjustments
        back_xmin, back_ymin, back_xmax, back_ymax = [0, 0, image_shape[0], image_shape[1]]
        img_xmin, img_ymin, img_xmax, img_ymax = self.box2d_diagonal
        x_clip = np.clip([img_xmin, img_xmax], back_xmin, back_xmax)
        y_clip = np.clip([img_ymin, img_ymax], back_ymin, back_ymax)
        truncated_box2d_diagonal = [x_clip[0], y_clip[0], x_clip[1], y_clip[1]]
        truncated_box2d = Box2D.from_box2d_diagonal(truncated_box2d_diagonal)
        box_size = self.get_box_size()
        if box_size <= 0:
            truncated_ratio = 0.0
        else:
            box2d_overlapping = box2d_overlapping_area(truncated_box2d_diagonal, self.box2d_diagonal)
            truncated_ratio = (box_size - box2d_overlapping) / max(box_size, 1e-6)
        if return_ratio:
            return truncated_box2d, truncated_ratio
        else:
            return truncated_box2d

    def get_box_center(self):
        x_min, y_min, x_max, y_max = self.box2d_diagonal
        return [int(0.5 * (x_min + x_max)), int(0.5 * (y_min + y_max))]

    def get_box_size(self):
        x_min, y_min, x_max, y_max = self.box2d_diagonal
        box_size = (y_max - y_min) * (x_max - x_min)
        return box_size

    def get_box_occlusion_ratio(self, other_boxes: np.ndarray) -> float:
        """
        :param other_boxes: All other 2D boxes in the image
        :return: Occlusion ratio for the current box
        """
        if len(other_boxes) == 0:
            return 0
        # Short-circuit when area is zero to avoid division by zero
        if self.shapely_box.area == 0:
            return 0
        box_union = None
        for box2 in other_boxes:
            box_inter = self.shapely_box.intersection(box2.shapely_box)
            if box_union is None:
                box_union = box_inter
            else:
                box_union = box_union.union(box_inter)
        if box_union is None:
            return 0
        occlusion_ratio = box_union.area / max(self.shapely_box.area, 1e-6)
        return occlusion_ratio
