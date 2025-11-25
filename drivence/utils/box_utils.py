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
    """
    对2D多边形的角点按顺时针方向排序，确保角点按顺时针顺序排列（适配多边形构建、碰撞检测等场景）。

    核心逻辑：
    1. 计算所有角点的几何中心点（质心）；
    2. 计算每个角点相对于质心的极角（与X轴正方向的夹角，范围[-π, π]）；
    3. 按极角升序排序，实现角点的顺时针排列（极角从小到大对应顺时针方向）。

    Args:
        corner (Union[np.ndarray, List[List[float]]]): 2D多边形角点集合，支持两种输入格式：
            - numpy数组：形状为 (N, 2)，N为角点数量（≥3，确保构成多边形），每行对应 [x, y] 坐标；
            - 列表：嵌套列表格式，如 [[x0,y0], [x1,y1], ..., [xn-1,yn-1]]，最终会转为numpy数组处理。

    Returns:
        np.ndarray: 按顺时针排序后的角点数组，形状为 (N, 2)，顺序为顺时针排列。
    """
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
    将LiDAR坐标系下的7维3D包围盒（中心+尺寸+朝向）转换为8个3D角点坐标。

    3D包围盒角点索引定义（参考函数内标注的空间位置）：
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    注：角点按上述顺序排列，基于包围盒中心、尺寸和朝向计算得出。

    核心逻辑：
    1. 生成单位立方体的8个角点模板（中心在原点，边长为1）；
    2. 按输入包围盒的尺寸缩放模板角点；
    3. 绕Z轴旋转缩放后的角点（根据包围盒朝向角）；
    4. 平移旋转后的角点到包围盒的中心坐标；
    5. 按指定格式（numpy/torch）返回结果。

    Args:
        boxes3d (Union[np.ndarray, torch.Tensor]): 3D包围盒数组，形状为 (N, 7) 或 (7,)：
            - 若为 (N, 7)：N为包围盒数量，每行为一个包围盒的参数 [x, y, z, dx, dy, dz, heading]；
            - 若为 (7,)：单个包围盒的参数，函数内部会自动扩展为 (1, 7) 处理；
            参数说明：
                x/y/z: 包围盒中心的3D坐标（LiDAR坐标系下）；
                dx/dy/dz: 包围盒在X/Y/Z轴方向的尺寸（长度）；
                heading: 包围盒绕Z轴的朝向角（单位：弧度，遵循右手定则）。
        return_type (str, 可选): 返回结果的数据类型，默认 "numpy"：
            - "numpy": 返回 numpy.ndarray；
            - "torch": 返回 torch.Tensor。

    Returns:
        Union[np.ndarray, torch.Tensor]: 8个3D角点坐标数组，形状为 (N, 8, 3) 或 (8, 3)：
            - 若输入为 (N, 7)：输出 (N, 8, 3)，N为包围盒数量，8为每个包围盒的角点数量，3为X/Y/Z坐标；
            - 若输入为 (7,)：输出 (8, 3)（单个包围盒的8个角点）；
            数据类型由 return_type 指定。
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
    """
    计算两个2D轴对齐包围盒（Axis-Aligned Bounding Box, AABB）的重叠面积。

    核心逻辑：
    1. 分别提取两个包围盒的左右（x_min/x_max）、上下（y_min/y_max）边界；
    2. 计算重叠区域的宽度（dx）：取两个包围盒右边界的最小值 - 左边界的最大值，若结果>0则存在水平重叠；
    3. 计算重叠区域的高度（dy）：取两个包围盒上边界的最小值 - 下边界的最大值，若结果>0则存在垂直重叠；
    4. 若水平和垂直方向均存在重叠（dx>0且dy>0），重叠面积=dx*dy；否则重叠面积为0。

    Args:
        box2d_a (Iterable): 第一个2D包围盒，格式为 [x_min, y_min, x_max, y_max]：
            - x_min: 左边界x坐标（最小x）；
            - y_min: 下边界y坐标（最小y）；
            - x_max: 右边界x坐标（最大x）；
            - y_max: 上边界y坐标（最大y）；
            要求 x_min < x_max 且 y_min < y_max（输入需确保合法性）。
        box2d_b (Iterable): 第二个2D包围盒，格式与 box2d_a 一致。

    Returns:
        float: 两个2D包围盒的重叠面积，非负数值（无重叠时返回0）。
    """
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
    """
    将Open3D的有向包围盒（OrientedBoundingBox）转换为8个3D角点的numpy数组。

    核心逻辑：直接调用Open3D有向包围盒的get_box_points()方法，获取其8个角点的坐标，
    并转换为numpy数组格式，便于后续3D碰撞检测、重叠计算等处理。

    Args:
        boxo3d (o3d.geometry.OrientedBoundingBox): Open3D的有向包围盒对象，
            包含包围盒的中心、旋转矩阵、尺寸等信息（支持任意姿态的包围盒，非轴对齐）。

    Returns:
        np.ndarray: 8个3D角点的坐标数组，形状为 (8, 3)，
            每行对应一个角点的 [x, y, z] 坐标（与Open3D get_box_points() 返回的顺序一致）。
    """
    corner_box = np.asarray(boxo3d.get_box_points())
    return corner_box


def corners3d_to_box_o3d(corners3d: np.ndarray) -> o3d.geometry.OrientedBoundingBox:
    """
    将3D角点数组转换为Open3D的有向包围盒（OrientedBoundingBox）对象。

    核心逻辑：利用Open3D的 `create_from_points` 静态方法，直接从输入的3D角点集合中拟合出
    最优的有向包围盒（非轴对齐，可适应角点对应的物体姿态），无需手动计算中心、旋转矩阵和尺寸。

    Args:
        corners3d (np.ndarray): 3D角点坐标数组，形状为 (8, 3)，
            每行对应一个角点的 [x, y, z] 坐标（角点需属于同一个3D物体的包围盒，顺序无强制要求）。

    Returns:
        o3d.geometry.OrientedBoundingBox: Open3D有向包围盒对象，
            包含包围盒的中心、旋转矩阵（姿态）、尺寸等信息，与输入角点集合的姿态和尺寸一致。
    """
    box3d = o3d.geometry.OrientedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(corners3d))
    return box3d

def covert_boxo3d_to_aligned_boxo3d(boxo3d: o3d.geometry.OrientedBoundingBox) -> o3d.geometry.OrientedBoundingBox:
    """Align an oriented bounding box with the ground plane using Euler angles."""
    # sciangle_0, sciangle_1, sciangle_2 = UtilsBox.get_euler_from_matrix(UtilsBox.get_box3d_R(box3d))
    """
    将Open3D有向包围盒（OrientedBoundingBox）与地面平面对齐，通过欧拉角调整包围盒姿态。

    核心逻辑：
    1. 从包围盒的旋转矩阵中提取欧拉角（Z-Y-X顺序，对应偏航、俯仰、滚转）；
    2. 根据欧拉角的数值范围判断包围盒姿态是否需要特殊调整；
    3. 通过逆旋转抵消原始姿态的欧拉角影响，再绕Z轴调整偏航角，最终使包围盒与地面对齐（X-Y平面平行）。

    Args:
        boxo3d (o3d.geometry.OrientedBoundingBox): 输入的Open3D有向包围盒对象，
            可能为任意姿态（非地面对齐），包含中心、旋转矩阵、尺寸等信息。

    Returns:
        o3d.geometry.OrientedBoundingBox: 与地面平面对齐后的有向包围盒对象，
            其X-Y平面与地面平行，仅保留绕Z轴的偏航角（用于碰撞检测、尺寸计算等场景）。
    """
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
    """
    从指定目录读取所有3D包围盒角点文件，解析为包含物体名称和角点坐标的字典列表。

    核心逻辑：
    1. 按自然排序读取目录下的所有包围盒文件；
    2. 解析每个文件名，提取物体名称（文件名格式约定：{obj_idx}_{obj_name}.xxx）；
    3. 读取文件内容，解析每行的3D角点坐标（x, y, z）；
    4. 将物体名称和角点坐标封装为字典，收集所有结果并返回。

    Args:
        bounding_box_dir (str): 包围盒文件所在目录的路径（绝对路径或相对路径），
            目录下的文件为文本格式，每行存储一个3D角点的 x/y/z 坐标。

    Returns:
        List[Dict[str, Any]]: 包含每个物体包围盒信息的字典列表，每个字典结构为：
            {
                "obj_name": str,  # 物体名称（从文件名中提取）
                "corners": np.ndarray  # 3D包围盒角点坐标数组，形状为 (N, 3)，N为角点数量（通常为8）
            }
    """
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
    """
    3D包围盒封装类，提供3D包围盒的构建、转换、属性计算等核心功能，
    支持从网格模型、角点数组、Open3D包围盒对象构建，适配多场景的包围盒处理需求。

    核心功能：
    1. 多源构建：支持从网格模型、3D角点数组、Open3D有向包围盒构建；
    2. 属性计算：获取包围盒范围、中心、底部中心、尺寸等关键属性；
    3. 格式转换：转换为鸟瞰图（BEV）2D包围盒、Open3D包围盒对象；
    4. 姿态调整：支持将包围盒与地面对齐（轴对齐优化）。
    """
    def __init__(self, box3d_corners=None):
        """
        初始化3D包围盒对象。

        Args:
            box3d_corners (Optional[np.ndarray], 可选): 3D包围盒的8个角点数组，形状为 (8, 3)，
                每行对应 [x, y, z] 坐标，默认 None（后续可通过 from_* 方法构建）。
        """
        self.box3d_corners = box3d_corners
        # self.calibration = None

    # def set_calibration(self,calibration):
    #     self.calibration = calibration

    def from_mesh(self, mesh_obj, align_to_axis=False):
        # get_minimal_oriented_bounding_box may yield a box that is not aligned with the z-axis
        # Open3D's align_bounding_box enforces alignment with the xy-plane, so apply manual corrections here
        # When align_to_axis=True the result is axis-aligned with z and orthogonal to the ground plane
        """
        从Open3D网格模型构建3D包围盒（基于模型的最小有向包围盒）。

        核心逻辑：
        1. 计算网格模型的最小有向包围盒（贴合模型形状，支持任意姿态）；
        2. 可选：将包围盒与地面对齐（通过 covert_boxo3d_to_aligned_boxo3d 方法）；
        3. 提取包围盒的8个角点，赋值给 self.box3d_corners。

        Args:
            mesh_obj (o3d.geometry.TriangleMesh): Open3D网格模型对象（需已正确初始化，包含几何信息）；
            align_to_axis (bool, 可选): 是否将包围盒与地面对齐（X-Y平面平行），默认 False：
                - True：调用 covert_boxo3d_to_aligned_boxo3d 调整姿态；
                - False：保留原始最小有向包围盒的姿态。
        """
        boxo3d = mesh_obj.get_minimal_oriented_bounding_box()
        if align_to_axis:
            boxo3d = covert_boxo3d_to_aligned_boxo3d(boxo3d)
        self.box3d_corners = box_o3d_to_corners3d(boxo3d)

    def from_box3d(self, box3d_corners):
        """
        从3D角点数组直接构建3D包围盒。

        Args:
            box3d_corners (np.ndarray): 3D包围盒的8个角点数组，形状为 (8, 3)，
                每行对应 [x, y, z] 坐标（角点顺序无强制要求，但需属于同一个3D包围盒）。
        """
        self.box3d_corners = box3d_corners

    def from_boxo3d(self, boxo3d: o3d.geometry.OrientedBoundingBox):
        """
        从Open3D有向包围盒对象构建3D包围盒。

        Args:
            boxo3d (o3d.geometry.OrientedBoundingBox): Open3D有向包围盒对象（包含中心、旋转矩阵、尺寸等信息）。
        """
        self.box3d_corners = box_o3d_to_corners3d(boxo3d)

    def get_o3d_box(self):
        """
        将当前3D包围盒转换为Open3D有向包围盒对象。

        Returns:
            o3d.geometry.OrientedBoundingBox: Open3D有向包围盒对象，与当前包围盒的角点、姿态一致。
        """
        boxo3d = corners3d_to_box_o3d(self.box3d_corners)
        return boxo3d

    def get_box_range(self, return_points=False):
        """
        计算3D包围盒的轴对齐范围（最小/最大X/Y/Z坐标）。

        Args:
            return_points (bool, 可选): 返回格式控制，默认 False：
                - False：返回6个数值（x_min, x_max, y_min, y_max, z_min, z_max）；
                - True：返回两个3D点（最小点 [x_min, y_min, z_min]，最大点 [x_max, y_max, z_max]）。

        Returns:
            包围盒的轴对齐范围，格式由 return_points 决定：
                - return_points=False：Tuple[float, float, float, float, float, float]；
                - return_points=True：Tuple[np.ndarray, np.ndarray]（形状均为 (3,)）。
        """
        x_min, y_min, z_min = np.min(self.box3d_corners, axis=0)
        x_max, y_max, z_max = np.max(self.box3d_corners, axis=0)
        if return_points:
            return (x_min, y_min, z_min), (x_max, y_max, z_max)
        else:
            return x_min, x_max, y_min, y_max, z_min, z_max

    # Keep the top-four corners (bird's-eye view)
    def to_box_bev(self):
        """
        将3D包围盒转换为鸟瞰图（BEV）的2D包围盒（仅保留顶部4个角点，按顺时针排序）。

        核心逻辑：
        1. 按Z轴坐标降序排序角点（筛选出Z值最大的4个顶部角点）；
        2. 提取顶部角点的X-Y坐标（忽略Z轴，投影到鸟瞰图平面）；
        3. 对2D角点按顺时针排序，返回排序后的结果。

        Returns:
            np.ndarray: 鸟瞰图2D包围盒的4个角点数组，形状为 (4, 2)，
                每行对应 [x, y] 坐标，按顺时针顺序排列。
        """
        box3d_corners = self.box3d_corners
        sorted_indices = np.argsort(box3d_corners[:, 2])[::-1]
        top_indices = sorted_indices[:4]
        top_corners = box3d_corners[top_indices][:, :2]
        top_corners = sort_corners_clockwise(top_corners)
        return top_corners

    def get_box_bottom_center(self):
        """
        计算3D包围盒底部中心的3D坐标（X-Y平面中心，Z轴取最小值）。

        Returns:
            np.ndarray: 底部中心坐标数组，形状为 (3,)，格式为 [x, y, z]，
                其中 x=(x_min+x_max)/2，y=(y_min+y_max)/2，z=z_min。
        """
        box_range = self.get_box_range()
        x_min, x_max, y_min, y_max, z_min, z_max = box_range
        bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
        return bottom_center

    def get_box_size(self):
        """
        计算3D包围盒的尺寸（高度、宽度、长度），基于Open3D包围盒的extent属性。

        Returns:
            Tuple[float, float, float]: 包围盒尺寸，格式为 (h, w, l)：
                - h: 高度（Z轴方向尺寸）；
                - w: 宽度（Y轴方向尺寸）；
                - l: 长度（X轴方向尺寸）。
        """
        o3d_box = self.get_o3d_box()
        x_, y_, z_ = np.asarray(o3d_box.extent)
        h, w, l = z_, y_, x_
        return h, w, l

    def get_box_center(self):
        """
        计算3D包围盒的几何中心（所有角点的平均坐标，或轴对齐范围的中点）。

        Returns:
            np.ndarray: 几何中心坐标数组，形状为 (3,)，格式为 [x, y, z]。
        """
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
    """
    将2D轴对齐包围盒的对角边界（左、上、右、下）转换为中心坐标+宽高格式。

    核心逻辑：
    1. 中心坐标 = 对角边界的中点（x中心=(左+右)/2，y中心=(上+下)/2）；
    2. 宽度 = 右边界 - 左边界（X轴方向跨度）；
    3. 长度 = 上边界 - 下边界（Y轴方向跨度）；
    4. 断言宽高均为正数，确保输入边界的合法性。

    Args:
        left (float): 包围盒左边界X坐标（最小X值）；
        top (float): 包围盒上边界Y坐标（最大Y值）；
        right (float): 包围盒右边界X坐标（最大X值）；
        bottom (float): 包围盒下边界Y坐标（最小Y值）；
        要求：left < right 且 bottom < top（输入需确保边界顺序合法）。

    Returns:
        tuple[float, float, float, float]: 转换后的包围盒参数，格式为 (x_center, y_center, width, length)：
            - x_center: 包围盒X轴中心坐标；
            - y_center: 包围盒Y轴中心坐标；
            - width: 包围盒X轴方向宽度（右-左）；
            - length: 包围盒Y轴方向长度（上-下）。
    """
    # Compute center, width, and length
    x_center = (left + right) / 2
    y_center = (top + bottom) / 2
    width = right - left
    length = top - bottom  # Length along the y-axis
    assert width > 0 and length > 0
    return x_center, y_center, width, length


def box2d_center_to_diagonal(x_center, y_center, width, length):
    """
    将2D轴对齐包围盒的“中心坐标+宽高”格式，转换为“对角边界（左、上、右、下）”格式。

    核心逻辑：
    1. 计算宽度和长度的一半（用于从中心向两侧扩展）；
    2. 左边界 = 中心X - 半宽，右边界 = 中心X + 半宽（X轴方向扩展）；
    3. 上边界 = 中心Y - 半长，下边界 = 中心Y + 半长（Y轴方向扩展，注释标注上边界为y_max、下边界为y_min，需匹配对应坐标系）；
    4. 直接返回扩展后的对角边界参数。

    Args:
        x_center (float): 包围盒X轴中心坐标；
        y_center (float): 包围盒Y轴中心坐标；
        width (float): 包围盒X轴方向宽度（必须为正数）；
        length (float): 包围盒Y轴方向长度（必须为正数）；
        要求：width > 0 且 length > 0（输入需确保尺寸合法）。

    Returns:
        tuple[float, float, float, float]: 转换后的对角边界参数，格式为 (left, top, right, bottom)：
            - left: 左边界X坐标（最小X值）；
            - top: 上边界Y坐标（最大Y值，即y_max）；
            - right: 右边界X坐标（最大X值）；
            - bottom: 下边界Y坐标（最小Y值，即y_min）。
    """
    # Compute left, top, right, and bottom
    half_width = width / 2
    half_length = length / 2
    left = x_center - half_width
    top = y_center - half_length  # y_max
    right = x_center + half_width
    bottom = y_center + half_length  # y_min
    return left, top, right, bottom



def box2d_corner_to_diagonal(box2d_corners):
    """
    将2D包围盒的角点数组转换为“中心坐标+宽高”格式（注：函数内存在恒假断言，运行时会触发AssertionError）。

    核心逻辑（理论上）：
    1. 计算所有角点X坐标的平均值作为包围盒中心X坐标；
    2. 计算所有角点Y坐标的平均值作为包围盒中心Y坐标；
    3. 计算X轴方向的最大差值（max_x - min_x）作为包围盒宽度；
    4. 计算Y轴方向的最大差值（max_y - min_y）作为包围盒长度；
    5. 存在恒假断言 `assert 1 == 2`，无论输入是否合法，都会触发断言失败。

    Args:
        box2d_corners (np.ndarray): 2D包围盒的角点数组，形状为 (N, 2)，
            N为角点数量（通常为4，对应四边形包围盒），每行对应 [x, y] 坐标。

    Returns:
        Tuple[float, float, float, float]: 理论上返回 (x_center, y_center, width, length)，
            但因断言恒假，实际运行时不会执行到返回语句。
    """
    x_center = np.mean(box2d_corners[:, 0])
    y_center = np.mean(box2d_corners[:, 1])
    width = np.max(box2d_corners[:, 0]) - np.min(box2d_corners[:, 0])
    length = np.max(box2d_corners[:, 1]) - np.min(box2d_corners[:, 1])
    assert 1 == 2
    return x_center, y_center, width, length


class Box2D(object):
    """
    2D包围盒封装类，支持轴对齐和旋转包围盒的构建、转换、碰撞检测、可视化等功能，
    基于 shapely 库实现几何运算，适配多场景的2D包围盒处理需求。

    核心功能：
    1. 多源构建：支持从角点数组、中心+宽高+偏航角、对角边界+偏航角构建；
    2. 几何运算：旋转、截断（适配图像边界）、相交判断、遮挡率计算；
    3. 属性获取：中心坐标、尺寸、角点、外接边界等；
    4. 可视化：支持通过 matplotlib 绘制包围盒。
    """
    def __init__(self):
        self.shapely_box = None
        self.box2d_diagonal = None  # (x_min,y_min,x_max,y_max) box2d_diagonal is the aligned box2d

    # Handles either rotated or axis-aligned corners
    @classmethod
    def from_corners(cls, corners):
        """
        从2D角点数组构建2D包围盒（支持轴对齐或旋转的包围盒）。

        核心逻辑：将角点数组转换为 shapely.Polygon 对象，自动计算其轴对齐外接边界。

        Args:
            corners (Union[np.ndarray, List[List[float]]]): 2D角点集合，格式为 (N, 2)，
                N为角点数量（通常为4，对应四边形），每行对应 [x, y] 坐标，角点需按顺时针/逆时针顺序排列。

        Returns:
            Box2D: 构建后的2D包围盒对象。
        """
        box2d = cls()
        # box2d.shapely_box = box
        box2d.shapely_box = Polygon(corners)
        box2d.box2d_diagonal = box2d.shapely_box.bounds  # (x_min,y_min,x_max,y_max)
        return box2d

    # box2d_center before rotation and the yaw to apply
    @classmethod
    def from_box2d_center(cls, box2d_center, yaw=0):  # # box2d (x_center,y_center,w (dx),l (dy),yaw)
        """
        从“中心坐标+宽高”格式构建2D包围盒，支持指定偏航角（旋转角度）。

        核心逻辑：
        1. 将“中心(x,y)、宽度w、长度l”转换为轴对齐的对角边界；
        2. 构建轴对齐的 shapely 包围盒；
        3. 按指定偏航角旋转包围盒（绕中心旋转）。

        Args:
            box2d_center (Tuple[float, float, float, float]): 包围盒参数，格式为 (x_center, y_center, w, l)：
                - x_center/y_center: 包围盒中心坐标；
                - w: X轴方向宽度（轴对齐时）；
                - l: Y轴方向长度（轴对齐时）；
            yaw (float, 可选): 偏航角（旋转角度），单位为弧度，默认0.0（轴对齐）：
                - 正值：逆时针旋转；
                - 负值：顺时针旋转。

        Returns:
            Box2D: 构建并旋转后的2D包围盒对象。
        """
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
        """
        从轴对齐的对角边界构建2D包围盒，支持指定偏航角（旋转角度）。

        核心逻辑：
        1. 从对角边界（x_min, y_min, x_max, y_max）构建轴对齐的 shapely 包围盒；
        2. 按指定偏航角旋转包围盒（绕中心旋转）。

        Args:
            box2d_diagonal (Tuple[float, float, float, float]): 轴对齐的对角边界，格式为 (x_min, y_min, x_max, y_max)，
                要求 x_min < x_max 且 y_min < y_max；
            yaw (float, 可选): 偏航角（旋转角度），单位为弧度，默认0.0（轴对齐）：
                - 正值：逆时针旋转；
                - 负值：顺时针旋转。

        Returns:
            Box2D: 构建并旋转后的2D包围盒对象。
        """
        shapely_box = box(*box2d_diagonal)
        box2d = cls()
        box2d.shapely_box = shapely_box
        box2d.box2d_diagonal = shapely_box.bounds
        box2d.rotate(yaw)
        return box2d

    def show(self, color="red", alpha=0.5, axis_equal=False):
        """
        通过 matplotlib 绘制当前2D包围盒。

        Args:
            color (str, 可选): 包围盒填充颜色，默认 "red"；
            alpha (float, 可选): 填充透明度（0-1），默认 0.5；
            axis_equal (bool, 可选): 是否设置坐标轴等比例，默认 False（避免形状失真时建议设为 True）。
        """
        from matplotlib import pyplot as plt
        x, y = self.get_exterior()
        plt.fill(x, y, color=color, alpha=alpha)
        if axis_equal:
            plt.axis("equal")

    def is_intersect(self, other_box2d):
        """
        判断当前包围盒与另一个2D包围盒是否相交（含边界接触、部分重叠、完全包含）。

        Args:
            other_box2d (Box2D): 待判断的另一个2D包围盒对象。

        Returns:
            bool: 相交返回 True，否则返回 False。
        """
        return self.shapely_box.intersects(other_box2d.shapely_box)

    def rotate(self, yaw=None):
        """
        绕包围盒中心旋转指定角度（偏航角）。

        说明：
        - 旋转方向：正值为逆时针旋转，负值为顺时针旋转；
        - 旋转后会更新 shapely_box 和 box2d_diagonal（重新计算轴对齐外接边界）。

        Args:
            yaw (float): 旋转角度（偏航角），单位为弧度。
        """
        # Positive angles represent counter-clockwise rotation; negative angles are clockwise
        self.shapely_box = affinity.rotate(self.shapely_box, yaw, origin="centroid", use_radians=True)

    def get_exterior(self):
        """
        获取包围盒外边界的x、y坐标数组（适配 matplotlib 绘制）。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 外边界坐标，格式为 (x_array, y_array)，
                每个数组的长度为 N+1（N为角点数量），最后一个点与第一个点重合（闭合多边形）。
        """
        return np.array(self.shapely_box.exterior.xy)

    def get_corners(self):
        """
        获取包围盒的2D角点数组（不含闭合重复的最后一个点）。

        Returns:
            np.ndarray: 角点数组，形状为 (N, 2)，N为角点数量（通常为4），每行对应 [x, y] 坐标。
        """
        return self.get_exterior().T[:-1]

    def truncate(self, image_shape, return_ratio=True):
        """
        将包围盒截断到图像边界内（去除超出图像的部分），返回截断后的包围盒和截断比例。

        核心逻辑：
        1. 计算包围盒与图像边界的交集（截断后的有效区域）；
        2. 计算截断比例（截断面积 / 原面积）；
        3. 返回截断后的包围盒和比例（或仅返回包围盒）。

        Args:
            image_shape (Tuple[int, int]): 图像尺寸，格式为 (height, width)，
                图像边界为 (0, 0) 到 (width, height)；
            return_ratio (bool, 可选): 是否返回截断比例，默认 True。

        Returns:
            Union[Tuple["Box2D", float], "Box2D"]:
                - return_ratio=True：返回 (truncated_box2d, truncated_ratio)，
                  truncated_ratio 为截断比例（0-1，0表示无截断，1表示完全截断）；
                - return_ratio=False：仅返回截断后的 Box2D 对象。
        """
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
        """
        获取包围盒轴对齐外接边界的中心坐标（整数型，适配图像像素坐标系）。

        Returns:
            List[int]: 中心坐标，格式为 [x_center, y_center]。
        """
        x_min, y_min, x_max, y_max = self.box2d_diagonal
        return [int(0.5 * (x_min + x_max)), int(0.5 * (y_min + y_max))]

    def get_box_size(self):
        """
        计算包围盒轴对齐外接边界的面积。

        Returns:
            float: 轴对齐外接边界的面积（非旋转包围盒的实际面积，仅为外接矩形面积）。
        """
        x_min, y_min, x_max, y_max = self.box2d_diagonal
        box_size = (y_max - y_min) * (x_max - x_min)
        return box_size

    def get_box_occlusion_ratio(self, other_boxes: np.ndarray) -> float:
        """
        计算当前包围盒被其他多个2D包围盒的遮挡率。

        核心逻辑：
        1. 计算当前包围盒与所有其他包围盒的交集区域的并集；
        2. 遮挡率 = 并集面积 / 当前包围盒面积（0表示无遮挡，1表示完全遮挡）。

        Args:
            other_boxes (List[Box2D]): 其他2D包围盒列表（需已初始化，含 shapely_box 属性）。

        Returns:
            float: 遮挡率（0.0-1.0），若当前包围盒面积为0或无其他包围盒，返回0.0。
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
