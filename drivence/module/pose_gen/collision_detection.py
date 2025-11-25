import numpy
from shapely.geometry import Polygon, Point


class CollisionDetector(object):
    def __init__(self):
        self.is_front_initial_obj = True

    def collision_detection(self, boxes_ori_corner: numpy.ndarray, boxes_inserted_corner: numpy.ndarray,
                            box_inserted_corner_now: numpy.ndarray) -> bool:
        """
        检测待插入目标与背景目标、已插入目标之间的 3D 碰撞（重叠），用于场景目标放置的合法性校验。

        该函数是场景生成过程中的核心碰撞校验接口，通过判断待插入目标的 3D 边界框与两类目标（背景固有目标、已插入自定义目标）的边界框是否重叠，
        避免目标在 3D 空间中相互穿透，确保场景的物理合理性（如车辆不会穿透道路障碍物、插入的多辆车之间无重叠）。
        碰撞判断依赖两个底层校验方法：`is_occluded_by_initial_obj`（与背景目标碰撞）和 `is_overlapped_with_inserted_obj`（与已插入目标碰撞），
        只要满足任一碰撞条件，即判定为碰撞。

        核心逻辑：
        1. 背景目标碰撞校验：检查待插入目标与所有背景固有目标的 3D 边界框是否重叠；
        2. 已插入目标碰撞校验：检查待插入目标与所有已放置的自定义目标的 3D 边界框是否重叠；
        3. 碰撞结果聚合：只要存在任一类型的碰撞（背景/已插入目标），返回 True；均无碰撞则返回 False。

        Args:
            boxes_ori_corner (np.ndarray): 背景固有目标的 3D 边界框角点数组，形状为 (M, 8, 3)，数据类型为 np.float32。
                - M: 背景目标的数量（如道路、建筑物、固有障碍物等）；
                - 8: 每个 3D 边界框的 8 个角点（长方体的 8 个顶点）；
                - 3: 每个角点的 X、Y、Z 三维坐标（需与其他边界框坐标系一致，如 LiDAR 坐标系）。
            boxes_inserted_corner (np.ndarray): 已插入自定义目标的 3D 边界框角点数组，形状为 (K, 8, 3)，数据类型为 np.float32。
                - K: 已插入目标的数量（如之前放置的车辆、行人等）；
                - 8/3: 与 boxes_ori_corner 的角点定义、坐标系完全一致；
                若暂无已插入目标，可传入形状为 (0, 8, 3) 的空数组。
            box_inserted_corner_now (np.ndarray): 当前待插入目标的 3D 边界框角点数组，形状为 (8, 3)，数据类型为 np.float32。
                - 8: 待插入目标边界框的 8 个角点；
                - 3: 每个角点的 X、Y、Z 三维坐标，需与前两个参数的坐标系严格一致（否则碰撞判断失效）。

        Returns:
            bool: 碰撞检测结果：
                - True：待插入目标与背景目标或已插入目标发生碰撞（边界框重叠），不允许插入；
                - False：待插入目标与所有目标均无碰撞，可合法插入场景。
        """
        is_collided_with_ini = self.is_occluded_by_initial_obj(boxes_ori_corner, box_inserted_corner_now)
        is_collided_with_peer = self.is_overlapped_with_inserted_obj(boxes_inserted_corner, box_inserted_corner_now)

        return is_collided_with_peer or is_collided_with_ini

    def is_overlapped_with_inserted_obj(self, boxes_ori_corner: numpy.ndarray,
                                        box_insert_corner: numpy.ndarray) -> bool:
        """
        检测当前待插入目标与所有已插入目标在 XY 平面的 2D 重叠，判断是否存在碰撞风险。

        该方法是 3D 碰撞检测的核心子逻辑，基于“XY 平面投影重叠”简化碰撞判断（适用于地面目标场景）：
        忽略 Z 轴高度信息，仅校验目标在水平面上的投影边界框是否重叠，若重叠则判定为潜在碰撞。
        该逻辑适用于车辆、行人等地面目标的场景插入校验，兼顾检测效率与准确性，避免目标在水平方向相互穿透。

        核心逻辑：
        1. 数据提取：从每个已插入目标的 3D 边界框角点中，提取前 4 个角点的 XY 平面坐标（代表水平投影的关键顶点）；
        2. 逐目标校验：遍历所有已插入目标，调用 `_is_overlapped_2d` 方法判断其与待插入目标的 XY 投影是否重叠；
        3. 结果判定：只要存在任一已插入目标与待插入目标的 2D 投影重叠，立即返回 True（碰撞），遍历结束无重叠则返回 False。

        Args:
            boxes_ori_corner (np.ndarray): 所有已插入目标的 3D 边界框角点数组，形状为 (K, 8, 3)，数据类型为 np.float32。
                - K: 已插入目标的数量（如之前放置的车辆、行人等）；
                - 8: 每个 3D 边界框的 8 个角点（长方体的 8 个顶点）；
                - 3: 每个角点的 X、Y、Z 三维坐标（需与待插入目标坐标系一致，如 LiDAR 坐标系）；
                若暂无已插入目标，传入形状为 (0, 8, 3) 的空数组时，直接返回 False。
            box_insert_corner (np.ndarray): 当前待插入目标的 3D 边界框角点数组，形状为 (8, 3)，数据类型为 np.float32。
                - 8: 待插入目标边界框的 8 个角点；
                - 3: 每个角点的 X、Y、Z 三维坐标，需与已插入目标的坐标系严格一致（否则投影重叠判断失效）。

        Returns:
            bool: 2D 重叠检测结果（对应 3D 碰撞风险）：
                - True：待插入目标与至少一个已插入目标的 XY 投影重叠，判定为存在碰撞；
                - False：待插入目标与所有已插入目标的 XY 投影均无重叠，判定为无碰撞。
        """
        is_collision = False
        for box_ori_corner in boxes_ori_corner:
            if self._is_overlapped_2d(box_ori_corner[0:4, 0:2], box_insert_corner[0:4, 0:2]):
                is_collision = True
                break
        return is_collision

    def is_occluded_by_initial_obj(self, boxes_ori_corner: numpy.ndarray, box_insert_corner: numpy.ndarray) -> bool:
        """
        检测待插入目标是否被背景固有目标遮挡或重叠，确保场景插入的物理合理性与视觉有效性。

        该方法结合「距离优先级」和「空间重叠校验」实现遮挡/重叠判断，核心逻辑适配激光雷达场景：
        1. 背景目标若比待插入目标更靠近激光雷达（LiDAR 中心），且在 Y 轴范围有重叠，则待插入目标会被背景遮挡（视为碰撞，不允许插入）；
        2. 若待插入目标与背景目标在 XY 平面投影直接重叠（无论距离），则判定为物理重叠（视为碰撞，不允许插入）。
        适用于避免待插入目标（如车辆）被背景障碍物（如建筑物、路沿）遮挡或穿透的场景。

        核心逻辑：
        1. 预处理：提取待插入目标的 XY 投影多边形、中心坐标及 Y 轴范围；
        2. 激光雷达中心定义：以 (0.0, 0.0) 为 LiDAR 原点（适配激光雷达坐标系）；
        3. 逐背景目标校验：
           a. 提取背景目标的 XY 投影多边形、中心坐标及 Y 轴范围；
           b. 距离判断：背景目标是否比待插入目标更靠近 LiDAR 中心；
           c. 1D 重叠：背景与待插入目标的 Y 轴范围是否重叠；
           d. 2D 重叠：背景与待插入目标的 XY 投影是否直接重叠；
           e. 判定规则：（距离更近 + Y 轴重叠）或（XY 直接重叠），满足任一则判定为遮挡/重叠。

        Args:
            boxes_ori_corner (np.ndarray): 背景固有目标的 3D 边界框角点数组，形状为 (M, 8, 3)，数据类型为 np.float32。
                - M: 背景目标的数量（如建筑物、路沿、固定障碍物等）；
                - 8: 每个 3D 边界框的 8 个角点（长方体的 8 个顶点）；
                - 3: 每个角点的 X、Y、Z 三维坐标（必须为激光雷达坐标系，与待插入目标一致）；
                若无背景目标，传入形状为 (0, 8, 3) 的空数组时，直接返回 False。
            box_insert_corner (np.ndarray): 当前待插入目标的 3D 边界框角点数组，形状为 (8, 3)，数据类型为 np.float32。
                - 8: 待插入目标边界框的 8 个角点；
                - 3: 每个角点的 X、Y、Z 三维坐标（激光雷达坐标系，需与背景目标严格一致）。

        Returns:
            bool: 遮挡/重叠检测结果：
                - True：待插入目标被背景目标遮挡或物理重叠，不允许插入；
                - False：待插入目标与背景目标无遮挡且无重叠，可合法插入。
        """
        is_collision = False
        corner_insert_polygon_points = box_insert_corner[0:4, 0:2]
        corner_insert_polygon = Polygon(corner_insert_polygon_points)
        corner_insert_center = corner_insert_polygon.centroid
        lidar_center = Point(0.0, 0.0)
        insert_y_range = [corner_insert_polygon_points.min(axis=0)[1], corner_insert_polygon_points.max(axis=0)[1]]

        for box_ori_corner in boxes_ori_corner:
            corner_ori_polygon_points = box_ori_corner[0:4, 0:2]
            corner_ori_polygon = Polygon(corner_ori_polygon_points)

            corner_ori_center = corner_ori_polygon.centroid

            ori_y_range = [corner_ori_polygon_points.min(axis=0)[1], corner_ori_polygon_points.max(axis=0)[1]]

            if lidar_center.distance(corner_insert_center) >= lidar_center.distance(
                    corner_ori_center) and self._is_overlapped_1d(ori_y_range,
                                                                  insert_y_range) or self._is_overlapped_2d(
                box_ori_corner[0:4, 0:2], box_insert_corner[0:4, 0:2]):
                is_collision = True
                break
        return is_collision

    def _is_overlapped_1d(self, interval1: list, interval2: list) -> bool:
        """
        检查两个一维区间是否存在重叠。

        核心逻辑：计算两个区间的重叠长度，若重叠长度大于 0 则判定为重叠，否则不重叠。
        一维区间格式为 [左边界, 右边界]，要求左边界 ≤ 右边界（输入需提前确保合法性）。

        Args:
            interval1 (list): 第一个一维区间，格式为 [start1, end1]，其中 start1 ≤ end1（数值型元素，如 int/float）；
            interval2 (list): 第二个一维区间，格式为 [start2, end2]，其中 start2 ≤ end2（数值型元素，如 int/float）。

        Returns:
            bool: 区间是否重叠的结果：
                - True：两个区间存在重叠（重叠长度 > 0）；
                - False：两个区间不重叠（无重叠或仅端点接触）。

        """
        overlap_length = min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
        if overlap_length > 0:
            return True
        return False

    def _is_overlapped_2d(self, box2d_ori_corner: numpy.ndarray, box2d_insert_corner: numpy.ndarray) -> bool:
        """
        检查两个2D包围盒是否存在重叠（含边界接触、部分重叠、完全包含）。

        核心逻辑：利用 shapely 的 Polygon 类，将两个2D包围盒的角点序列转换为多边形对象，
        通过多边形的 intersects 方法判断是否存在几何交集，实现2D重叠检测。

        Args:
            box2d_ori_corner (numpy.ndarray): 原始2D包围盒的角点数组，形状为 (4, 2)，
                每行对应一个角点的 (x, y) 坐标，角点需按顺时针或逆时针顺序有序排列（确保能构成闭合多边形）；
            box2d_insert_corner (numpy.ndarray): 待插入2D包围盒的角点数组，格式与 box2d_ori_corner 一致，
                需满足 (4, 2) 形状和角点有序排列要求。

        Returns:
            bool: 两个2D包围盒是否重叠的结果：
                - True：存在重叠（含部分重叠、完全包含、边界接触）；
                - False：无任何重叠。
        """
        corner_ori_polygon_points = [list(corner) for corner in box2d_ori_corner]
        corner_insert_polygon_points = [list(corner) for corner in box2d_insert_corner]

        corner_ori_polygon = Polygon(corner_ori_polygon_points)
        corner_insert_polygon = Polygon(corner_insert_polygon_points)

        return corner_ori_polygon.intersects(corner_insert_polygon)
