
import yaml
import os
import re
from easydict import EasyDict as edict
import importlib

from drivence.utils.path_utils import get_project_root_dir


class BaseConfig:
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_yaml(cls, file_path):

        with open(file_path, 'r', encoding="utf-8") as file:
            # lidar_config = yaml.safe_load(file)
            config = edict(yaml.safe_load(file))
        variables = cls.extract_variables(config)
        # 假设项目根目录是已知的，作为上下文传递给路径解析
        context = {}
        for key, value in variables.items():
            context[key] = cls.get_nested_attr(config, key)
        # 根据上下文解析路径
        resolved_config = cls.resolve_paths(config, context)
        return cls(resolved_config)

    # 递归解析路径变量
    @classmethod
    def resolve_paths(cls, config, context=None):
        if context is None:
            context = {}

        # 判断当前元素是字典、列表还是普通字符串
        if isinstance(config, edict):  # 使用 EasyDict 进行判断
            for key, value in config.items():
                # 递归处理字典中的每个值
                config[key] = cls.resolve_paths(value, context)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                # 递归处理列表中的每个元素
                config[i] = cls.resolve_paths(item, context)
        elif isinstance(config, str):
            while True:
                # 查找路径中的变量并替换
                match = re.search(r"\${(.*?)}", config)
                if not match:
                    break
                var_name = match.group(1)
                # 获取上下文中的变量值（例如project_structure.root_dir）
                var_value = context.get(var_name, "")
                if var_value:
                    config = config.replace(f"${{{var_name}}}", var_value)
            return config
        return config

    # 获取路径中的变量（可以设置为手动传入或自动提取）
    @classmethod
    def extract_variables(cls, config):
        variables = {}

        def extract(config, prefix=""):
            if isinstance(config, edict):
                for key, value in config.items():
                    new_prefix = f"{prefix}{key}" if prefix else key
                    extract(value, f"{new_prefix}.")
            elif isinstance(config, list):
                for item in config:
                    extract(item, prefix)
            elif isinstance(config, str):
                matches = re.findall(r"\${(.*?)}", config)
                for match in matches:
                    if match not in variables:
                        variables[match] = None  # 存储所有路径中的变量
            return variables

        variables = extract(config)
        return variables

    @classmethod
    def get_nested_attr(cls, obj, attr_path):
        attributes = attr_path.split(".")
        for attr in attributes:
            obj = getattr(obj, attr)
        return obj


class ConfigLoader:
    @staticmethod
    def load_yaml(file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)


class BaseConfigParser(object):
    def __init__(self, file_path):
        self.config = self.from_yaml(file_path)

    def from_yaml(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            yaml_config = yaml.safe_load(file)
        return yaml_config

    def get_wrapped_config(self):
        return self.key_wrapper(self.config)

    def get_config(self):
        return self.config

    def key_wrapper(self, config):
        new_config = {
            list(config.keys())[0]: config
        }
        return new_config


class PathConfigParser(BaseConfigParser):
    def __init__(self, file_path):
        self.config = self.from_yaml(file_path)

    def get_config(self):
        return self.config

    def from_yaml(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
        converted_config = self._convert_configs(config)
        return converted_config

    def _convert_configs(self, data, key_to_convert="config"):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == key_to_convert and isinstance(value, str) and '.' in value:
                    # 仅转换特定键下的字符串
                    init_config_path = os.path.splitext(value)[0]
                    config_extension = os.path.splitext(value)[1]
                    config_path = init_config_path.replace('.',
                                                           "/")  # D:/Python/pythonHome/Drivence\drivence/configs/sensors/camera/camera_blender.yml
                    data[key] = config_path + config_extension
                else:
                    data[key] = self._convert_configs(value, key_to_convert)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = self._convert_configs(item, key_to_convert)

        return data


class DatasetConfigParser(BaseConfigParser):
    def __init__(self, file_path):
        super().__init__(file_path)  # 每次类只会读取一次yml文件
        self.file_path = file_path

    def get_config(self, dataset_name):
        config = self.config  # 保证每次获取的都是最新的config，一定需要拼接成绝对路径
        config[dataset_name]["root_dir"] = os.path.join(get_project_root_dir(), config[dataset_name]["root_dir"])
        return config[dataset_name]

    # def from_yaml(self, file_path, dataset_name):
    #     with open(file_path, 'r', encoding="utf-8") as file:
    #         config = yaml.safe_load(file)
    #     return config[dataset_name]

    def get_wrapped_config(self, dataset_name):
        return self.key_wrapper(self.get_config(dataset_name))


class AssetsConfigParser(PathConfigParser):
    def __init__(self, file_path):
        super().__init__(file_path)


class ProjectStructureConfigParser(BaseConfigParser):
    def __init__(self, file_path):
        super().__init__(file_path)

    def get_config(self):
        return self.config

    def from_yaml(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            # lidar_config = yaml.safe_load(file)
            config = edict(yaml.safe_load(file))
        variables = self.extract_variables(config)
        # 假设项目根目录是已知的，作为上下文传递给路径解析
        context = {}
        for key, value in variables.items():
            context[key] = self.get_nested_attr(config, key)
        # 根据上下文解析路径
        resolved_config = self.resolve_paths(config, context)
        return resolved_config

    # 递归解析路径变量
    def resolve_paths(self, config, context=None):
        if context is None:
            context = {}

        # 判断当前元素是字典、列表还是普通字符串
        if isinstance(config, edict):  # 使用 EasyDict 进行判断
            for key, value in config.items():
                # 递归处理字典中的每个值
                config[key] = self.resolve_paths(value, context)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                # 递归处理列表中的每个元素
                config[i] = self.resolve_paths(item, context)
        elif isinstance(config, str):
            while True:
                # 查找路径中的变量并替换
                match = re.search(r"\${(.*?)}", config)
                if not match:
                    break
                var_name = match.group(1)
                # 获取上下文中的变量值（例如project_structure.root_dir）
                var_value = context.get(var_name, "")
                if var_value:
                    config = config.replace(f"${{{var_name}}}", var_value)
            return config
        return config

    # 获取路径中的变量（可以设置为手动传入或自动提取）
    def extract_variables(self, config):
        variables = {}

        def extract(config, prefix=""):
            if isinstance(config, edict):
                for key, value in config.items():
                    new_prefix = f"{prefix}{key}" if prefix else key
                    extract(value, f"{new_prefix}.")
            elif isinstance(config, list):
                for item in config:
                    extract(item, prefix)
            elif isinstance(config, str):
                matches = re.findall(r"\${(.*?)}", config)
                for match in matches:
                    if match not in variables:
                        variables[match] = None  # 存储所有路径中的变量
            return variables

        variables = extract(config)
        return variables

    def get_nested_attr(self, obj, attr_path):
        attributes = attr_path.split(".")
        for attr in attributes:
            obj = getattr(obj, attr)
        return obj


def dynamic_import(class_path, class_name):
    module = importlib.import_module(class_path)
    return getattr(module, class_name)


def build_component_dict(config_data):
    component_dict = {}

    def process_config(config, prefix=""):
        if isinstance(config, dict):
            for key, value in config.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if key == "classpath" and "classname" in config:
                    # 动态加载类
                    class_instance = dynamic_import(config["classpath"], config["classname"])
                    component_dict[config["name"]] = class_instance
                elif isinstance(value, dict):
                    # 递归处理子字典
                    process_config(value, full_key)
                elif isinstance(value, list):
                    # 处理列表中的每个元素
                    for idx, item in enumerate(value):
                        process_config(item, f"{full_key}[{idx}]")
        elif isinstance(config, list):
            for idx, item in enumerate(config):
                process_config(item, f"{prefix}[{idx}]")

    process_config(config_data)
    return component_dict


def get_class_dict(project_config_path):
    # 加载project.yml
    project_config = ConfigLoader.load_yaml(project_config_path)
    _class_dict = build_component_dict(project_config)
    return _class_dict
