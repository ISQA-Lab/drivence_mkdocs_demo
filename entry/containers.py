import yaml
import os
from dependency_injector import containers, providers

from configs.loader import ProjectStructureConfigParser, get_class_dict, PathConfigParser, DatasetConfigParser, AssetsConfigParser
from drivence.utils.path_utils import get_project_root_dir


class DataGenContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    root_dir = get_project_root_dir()
    project_structure_config_path = os.path.join(root_dir, "configs/project_structure.yml")
    project_structure_config = providers.Singleton(ProjectStructureConfigParser,
                                                   file_path=project_structure_config_path)
    project_config_path = project_structure_config().get_config()["project_structure"]["configs"]["project_config"]

    _class_dict = get_class_dict(project_config_path)

    project_config = providers.Singleton(PathConfigParser, file_path=project_config_path)
    lidar_config_path = os.path.join(root_dir,
                                     project_config().get_config()["project"]["data_gen"]["sensor"]["lidar"]["config"])
    lidar_config = providers.Singleton(PathConfigParser, file_path=lidar_config_path)

    dataset_config_path = os.path.join(root_dir, project_config().get_config()["project"]["dataset"]["config"])
    dataset_config_name = project_config().get_config()["project"]["dataset"]["name"]
    dataset_config = providers.Singleton(DatasetConfigParser, file_path=dataset_config_path)

    assets_config_path = os.path.join(root_dir, project_config().get_config()["project"]["assets"]["config"])
    assets_config = providers.Singleton(AssetsConfigParser, file_path=assets_config_path)

    intensity_config_provider = None
    try:
        intensity_config_path_str = project_config().get_config()["project"]["data_gen"]["intensity"]["config"]
        intensity_config_path = os.path.join(root_dir, intensity_config_path_str)
        intensity_config_provider = providers.Singleton(PathConfigParser, file_path=intensity_config_path)
    except (KeyError, TypeError):
        pass

    config.from_dict(project_structure_config().get_wrapped_config())
    config.from_dict(project_config().get_wrapped_config())
    config.from_dict(lidar_config().get_wrapped_config())
    config.from_dict(dataset_config().get_wrapped_config(dataset_config_name))
    config.from_dict(assets_config().get_wrapped_config())
    if intensity_config_provider is not None:
        config.from_dict(intensity_config_provider().get_wrapped_config())
    intensity_config = intensity_config_provider

    lidar_handle_occ_strategy = providers.Singleton(_class_dict.get("lidar_occlusion_handler"))
    virtual_lidar = providers.Singleton(_class_dict.get("virtual_lidar"),
                                         lidar_config=config.lidar,
                                         occlusion_handler=lidar_handle_occ_strategy())

    lidar_label_generator = providers.Singleton(_class_dict.get("lidar_label_generator"))

    shapenet_loader = providers.Singleton(_class_dict.get("shapenet_loader"), config.shapenet)

    road_split = providers.Singleton(
        _class_dict.get("road_split"),
        img_height=1080,
        img_width=1920,
    )


if __name__ == "__main__":
    container = DataGenContainer()
    container.virtual_lidar()
