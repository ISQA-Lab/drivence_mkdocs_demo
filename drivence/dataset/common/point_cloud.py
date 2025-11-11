import numpy as np
import open3d as o3d

from drivence.utils.pc_utils import PointCloud as BasePointCloud

class PointCloud(BasePointCloud):
    # @Test status: Completed
    def __init__(self, point_cloud: np.ndarray=None,reflection: np.ndarray=None):
        self.point_cloud = None
        self.reflection = None
   
    # @Test status: uncompleted (pc-only: no dataset label dependency)
    def visualize_with_label(self, label=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        o3d.visualization.draw_geometries([pcd])
       