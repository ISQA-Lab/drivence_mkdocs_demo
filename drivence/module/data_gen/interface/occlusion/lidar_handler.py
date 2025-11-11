from abc import abstractmethod
from typing import List

import open3d as o3d
import numpy as np


# -: 雷达
# O：物体
# []：阴影mesh
# -[center2obj_pyramid_mesh]O[obj_shadow_mesh]


#     -   雷达
#    ***  -和o是雷达物体间的视锥体 center2obj_pyramid_mesh
#   ooooo  物体
#  +*****+  
# +++++++++ +和o是阴影视锥体 obj_shadow_mesh

class LidarOcclusionHandler:
    def __init__(self):
        self.object_occluded_point_threshold = 10
        self.lidar_position = None

    def handle_occlusion_batch(self, pc_bg: np.ndarray, pc_obj: np.ndarray,
                               mesh_obj_list: List[o3d.geometry.TriangleMesh], return_mask=False,
                               labels_bg: np.ndarray = None, labels_obj: np.ndarray = None,
                               return_bg_keep_mask=False):
        """
        处理点云遮挡（基类接口）
        
        Args:
            pc_bg: 背景点云
            pc_obj: 物体点云
            mesh_obj_list: 物体网格列表
            return_mask: 是否返回完整的keep_mask
            labels_bg: 背景标签
            labels_obj: 物体标签
            return_bg_keep_mask: 是否返回背景点的保留掩码（用于正确映射intensity）
        """
        return None

    # 后面如果需要计算点云的mask，可以在外面使用循环，逐个调用handle_occlusion
    # pc_obj 和 mesh_obj 是一对一关系
    def handle_occlusion(self, pc_bg: np.ndarray, pc_obj: np.ndarray, mesh_obj: o3d.geometry.TriangleMesh,
                         return_mask=False, labels_bg: np.ndarray = None, labels_obj: np.ndarray = None):
        return None
