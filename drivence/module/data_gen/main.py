import os

from drivence.dataset.common.dataframe_factory import DataFrameFactory

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import math
from entry.containers import DataGenContainer
from drivence.entity.scene_info import SceneInfo

from drivence.utils.box_utils import box_o3d_to_corners3d, covert_boxo3d_to_aligned_boxo3d


def lidar_pipeline(ray_casting_lidar, label_lidar_gen, lidar_scene_info, shapenet_loader, dataset_config, intensity_config=None):
    """
    LiDAR data generation pipeline with unified output handling.
    """
    lidar = ray_casting_lidar
    lidar_label_generator = label_lidar_gen
    
    bounding_box_dicts, car_list = get_mesh_and_bounding_box(lidar_scene_info, shapenet_loader,
                                                             lidar.yaw_rotation_counterclockwise)

    dataset_type = dataset_config.get_config("source_dataset").get("dataset_type", "")
    dataset_output_type = dataset_config.get_config("source_dataset").get("dataset_output_type", "")
    
    dataset = DataFrameFactory.get_data_frame(dataset_type, lidar_scene_info.bg_index, dataset_config.get_config("source_dataset"))
    dataset_output = DataFrameFactory.get_output_data_frame(dataset_type, dataset_output_type, lidar_scene_info.bg_index, dataset_config.get_config("generated_dataset"))
    
    labels = dataset.get_label()
    calib_info = dataset.get_calibration()
    background_pointcloud = dataset.get_point_cloud()
    pc_bg = background_pointcloud.point_cloud
    pc_bg_intensity = background_pointcloud.reflection

    from lidar_core.occlusion.semantic_label_handler import create_semantic_label_handler
    semantic_handler = create_semantic_label_handler(dataset_type)
    original_semantic_labels = semantic_handler.get_original_semantic_labels(dataset)

    pcd_obj_numpy, instance_mask = lidar.simulate_all(car_list, return_rays=False, return_instance_mask=True)

    semantic_labels = getattr(lidar_scene_info, "semantic_labels", None)
    obj_names = getattr(lidar_scene_info, "obj_names", None)

    vehicle_semantic_labels = semantic_handler.generate_vehicle_semantic_labels(
        pcd_obj_numpy, instance_mask=instance_mask, semantic_labels=semantic_labels
    )

    pc_remain, semantic_labels_remain, bg_keep_mask, bg_remain_count, intensity_bg_remain, instance_mask_remain = ray_casting_lidar.handle_occlusion(
        pc_bg=pc_bg, pc_obj=pcd_obj_numpy, mesh_obj_list=car_list,
        labels_bg=original_semantic_labels, labels_obj=vehicle_semantic_labels,
        intensity_bg=pc_bg_intensity,
        instance_mask_obj=instance_mask,
        return_bg_keep_mask=True
    )

    mix_labels = None
    if labels is not None and calib_info is not None:
        mix_labels = lidar_label_generator.generate_labels(labels, bounding_box_dicts, calib_info, lidar_scene_info)

    if semantic_labels_remain is not None:
        fused_semantic_labels = semantic_labels_remain
    else:
        if original_semantic_labels is not None and bg_keep_mask is not None:
            bg_labels = original_semantic_labels[bg_keep_mask]
        else:
            bg_labels = np.zeros(bg_remain_count, dtype=np.uint32)

        obj_remain_count = len(pc_remain) - bg_remain_count
        if obj_remain_count > 0:
            if instance_mask_remain is not None and semantic_labels is not None:
                obj_labels = np.empty(obj_remain_count, dtype=np.uint32)
                for i, inst in enumerate(instance_mask_remain):
                    if 0 <= inst < len(semantic_labels):
                        obj_labels[i] = semantic_labels[inst]
                    else:
                        obj_labels[i] = semantic_handler.vehicle_label_id
            else:
                obj_labels = np.full(obj_remain_count, semantic_handler.vehicle_label_id, dtype=np.uint32)
        else:
            obj_labels = np.empty(0, dtype=np.uint32)

        fused_semantic_labels = np.concatenate([bg_labels, obj_labels], axis=0)

    from drivence.utils.pc_utils import create_fused_pointcloud_with_intensity
    fused_pointcloud = create_fused_pointcloud_with_intensity(
        pc_remain, intensity_bg_remain, pcd_obj_numpy, instance_mask_remain, obj_names, 
        bg_keep_mask=None,
        bg_remain_count=bg_remain_count,
        intensity_config=intensity_config
    )

    dataset_output.set(
        label=mix_labels, 
        calib=calib_info, 
        point_cloud=fused_pointcloud
    )
    semantic_handler.set_semantic_labels_to_output(dataset_output, fused_semantic_labels)
    dataset_output.save()


def get_mesh_and_bounding_box(lidar_scene_info, shapenet_loader, yaw_rotation_counterclockwise):
    bounding_box_dicts = []
    car_list = []
    
    from drivence.assets.assets_utils import load_mesh_from_vehicle
    
    for vehicle in lidar_scene_info.vehicles:
        car_mesh = load_mesh_from_vehicle(vehicle, shapenet_loader)
        
        car_mesh.translate(vehicle.location)
        rz_radians = math.radians(vehicle.rotation)
        if not yaw_rotation_counterclockwise:
            rz_radians = 2 * np.pi - rz_radians
        RZ = car_mesh.get_rotation_matrix_from_xyz((0, 0, rz_radians))
        car_mesh.rotate(RZ)
        car_list.append(car_mesh)

        obj_inserted_box3d = car_mesh.get_minimal_oriented_bounding_box()
        obj_box3d_adjusted = covert_boxo3d_to_aligned_boxo3d(obj_inserted_box3d)
        obj_box3d_corners = box_o3d_to_corners3d(obj_box3d_adjusted)
        bounding_box_dicts.append({"obj_name": vehicle.obj_name, "corners": np.array(obj_box3d_corners)})

    return bounding_box_dicts, car_list


def get_lidar_scene_info():
    ...


def run(task_name, lidar_scene_info):
    print("task_name:", task_name)
    if isinstance(lidar_scene_info, str) or isinstance(lidar_scene_info, dict):
        lidar_scene_info = SceneInfo(lidar_scene_info)

    container = DataGenContainer()
    dataset_config = container.dataset_config()
    
    intensity_config = None
    try:
        if container.intensity_config is not None:
            intensity_config_obj = container.intensity_config()
            intensity_config = intensity_config_obj.get_config()
    except (AttributeError, KeyError, TypeError):
        pass
 
    ray_casting_lidar = container.virtual_lidar()
    label_lidar_gen = container.lidar_label_generator()
    shapenet_loader = container.shapenet_loader()

    lidar_pipeline(ray_casting_lidar, label_lidar_gen, lidar_scene_info, shapenet_loader, dataset_config, intensity_config)


if __name__ == '__main__':
    pass
