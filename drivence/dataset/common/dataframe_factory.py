from drivence.dataset.semantickitti.dataset import (
    SemanticKITTIDetectionDataFrame,
    SemanticKITTIDetectionOutputDataFrame,
)


class DataFrameFactory:
    # 创建DataFrame的工厂类
    @staticmethod
    def get_data_frame(dataset_type, frame_id, dataset_config: dict):
        if dataset_type == "SemanticKITTI":
            return SemanticKITTIDetectionDataFrame(frame_id, dataset_config)
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    @staticmethod
    def get_output_data_frame(dataset_type, dataset_output_type, frame_id, dataset_config: dict):
        if dataset_type == "SemanticKITTI":
            return SemanticKITTIDetectionOutputDataFrame(frame_id, dataset_config)
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    # module_name, class_name = dataset_config["data_frame"].rsplit(".", 1)
    # import importlib
    # module = importlib.import_module(module_name)
    # clazz = getattr(module, class_name)
    # return clazz(frame_id, dataset_config)
