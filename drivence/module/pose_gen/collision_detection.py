import numpy
from shapely.geometry import Polygon, Point


class CollisionDetector(object):
    def __init__(self):
        self.is_front_initial_obj = True

    def collision_detection(self, boxes_ori_corner: numpy.ndarray, boxes_inserted_corner: numpy.ndarray,
                            box_inserted_corner_now: numpy.ndarray) -> bool:
        """
        Detect collisions between objects.
        
        Args:
            boxes_ori_corner: 3D bounding box corners of background objects
            boxes_inserted_corner: 3D bounding box corners of already inserted objects
            box_inserted_corner_now: 3D bounding box corners of current object to insert
        
        Returns:
            True if collision detected
        """
        is_collided_with_ini = self.is_occluded_by_initial_obj(boxes_ori_corner, box_inserted_corner_now)
        is_collided_with_peer = self.is_overlapped_with_inserted_obj(boxes_inserted_corner, box_inserted_corner_now)

        return is_collided_with_peer or is_collided_with_ini

    def is_overlapped_with_inserted_obj(self, boxes_ori_corner: numpy.ndarray,
                                        box_insert_corner: numpy.ndarray) -> bool:
        """Check if current object overlaps with already inserted objects."""
        is_collision = False
        for box_ori_corner in boxes_ori_corner:
            if self._is_overlapped_2d(box_ori_corner[0:4, 0:2], box_insert_corner[0:4, 0:2]):
                is_collision = True
                break
        return is_collision

    def is_occluded_by_initial_obj(self, boxes_ori_corner: numpy.ndarray, box_insert_corner: numpy.ndarray) -> bool:
        """Check if inserted object is occluded by or overlaps with background objects."""
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
        """Check if two 1D intervals overlap."""
        overlap_length = min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
        if overlap_length > 0:
            return True
        return False

    def _is_overlapped_2d(self, box2d_ori_corner: numpy.ndarray, box2d_insert_corner: numpy.ndarray) -> bool:
        """Check if two 2D boxes overlap."""
        corner_ori_polygon_points = [list(corner) for corner in box2d_ori_corner]
        corner_insert_polygon_points = [list(corner) for corner in box2d_insert_corner]

        corner_ori_polygon = Polygon(corner_ori_polygon_points)
        corner_insert_polygon = Polygon(corner_insert_polygon_points)

        return corner_ori_polygon.intersects(corner_insert_polygon)
