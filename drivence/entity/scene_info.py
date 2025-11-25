import copy
import json
from typing import List

vehicle_info_dict = {
    "obj_index": "",
    "obj_name": "",
    "location": [],
    "rotation": [],
    "scale": []
}


def generate_vehicle_infoes(obj_index, obj_name, location, rotation, scale):
    """
    批量生成多辆车辆的信息字典列表，按输入参数顺序一一对应封装车辆核心属性。

    该函数用于场景生成、仿真等批量处理场景，通过遍历输入的批量参数，调用单车辆信息生成函数，
    生成结构统一的车辆信息列表，便于后续网格加载、姿态变换、场景插入等流程统一调用。

    核心逻辑：
    1. 遍历输入的批量参数（索引、名称、位置等），确保每组参数一一对应；
    2. 对每一组参数，调用 `generate_vehicle_info` 生成单辆车的标准化信息字典；
    3. 收集所有车辆信息字典，返回统一格式的列表。

    Args:
        obj_index: 车辆唯一索引列表，用于区分不同车辆。
        obj_name: 车辆模型名称列表，需与网格资产加载器中的模型名称一致。
        location: 车辆3D位置列表，每辆车的坐标格式为 [x, y, z]（单位：米），
        rotation: 车辆旋转信息列表，每辆车的旋转格式为 [rx, ry, rz]（欧拉角，单位：度），
        scale: 车辆缩放信息列表

    Returns:
        vehicle_infoes: 批量车辆信息字典列表，包含 "obj_index"、"obj_name"、"location"、"rotation"、"scale" 五个字段，
    """
    vehicle_infoes = []
    for i in range(len(obj_index)):
        vehicle_info = generate_vehicle_info(obj_index[i], obj_name[i], location[i], rotation[i], scale[i])
        vehicle_infoes.append(vehicle_info)
    return vehicle_infoes


def generate_vehicle_info(obj_index, obj_name, location, rotation, scale):
    vehicle_info = copy.deepcopy(vehicle_info_dict)
    vehicle_info["obj_index"] = obj_index
    vehicle_info["obj_name"] = obj_name
    vehicle_info["location"] = location
    vehicle_info["rotation"] = rotation
    vehicle_info["scale"] = scale
    return vehicle_info


class Vehicle(object):
    """
    车辆实体类，封装单辆车辆的核心属性（索引、名称、姿态、尺度、语义标签等），
    支持属性序列化（字典/JSON 格式），为场景生成、点云仿真、数据存储等流程提供标准化车辆对象。

    核心特性：
    1. 严格校验输入信息格式，确保属性合法性；
    2. 封装车辆关键属性，支持默认值填充（如未指定名称时自动生成）；
    3. 提供 `to_dict`/`to_json` 方法，支持属性序列化，便于存储和传输；
    4. 兼容语义标签、模型路径等扩展属性，适配多场景需求。
    """

    def __init__(self, info: Dict):
        """
        初始化车辆对象，从输入字典中解析并校验核心属性。

        Args:
            info (Dict): 车辆信息字典，必须包含以下关键字段，支持可选扩展字段：
                - 必选字段：
                  - "obj_index": 车辆唯一索引（int/str 类型，如 0、"001"）；
                  - "location": 车辆 3D 位置坐标（列表/数组类型，格式 [x, y, z]，单位：米）；
                  - "rotation": 车辆旋转角度（int/float 类型，默认绕 Z 轴偏航角，单位：度）；
                  - "scale": 车辆缩放比例（int/float 类型，统一缩放系数）；
                - 可选字段：
                  - "obj_name": 车辆模型名称（str 类型，如 "car_sedan_01"，未指定时自动生成）；
                  - "semantic_label": 车辆语义标签（int 类型，如 10 对应车辆类别，未指定时为 None）；
                  - 其他扩展字段：如 "obj_path"（后续可通过实例属性赋值添加）。
        """
        # 校验输入信息类型（必须为字典）
        if not isinstance(info, dict):
            assert 1 == 2, "info is not a dict"
        self.obj_index = info["obj_index"]
        if "obj_name" in info:
            self.obj_name = info["obj_name"]
        else:
            self.obj_name = "Car_{}".format(self.obj_index)
        self.location = info["location"]
        self.rotation = info["rotation"]
        self.scale = info["scale"]
        self.obj_path = None
        # 支持语义标签
        if "semantic_label" in info:
            self.semantic_label = info["semantic_label"]
        else:
            self.semantic_label = None

    def to_dict(self) -> Dict[str, Union[int, str, List[float]]]:
        """
        将车辆对象的属性序列化为字典（与 `to_json` 功能一致，兼容不同调用习惯）。

        Returns:
            Dict[str, Union[int, str, List[float]]]: 车辆属性字典，包含所有非 None 字段，
                数值型字段统一转为 float/int，列表字段转为 float 列表，确保格式标准化。
        """
        return self.to_json()

    def to_json(self) -> Dict[str, Union[int, str, List[float]]]:
        """
        将车辆对象的属性序列化为 JSON 兼容的字典（支持序列化存储或网络传输）。

        Returns:
            Dict[str, Union[int, str, List[float]]]: JSON 兼容的车辆属性字典，字段说明：
                - "obj_index": 车辆索引（int 类型）；
                - "obj_name": 车辆模型名称（str 类型）；
                - "location": 3D 位置坐标（List[float] 类型，格式 [x, y, z]）；
                - "rotation": 旋转角度（float 类型）；
                - "scale": 缩放比例（float 类型）；
                - "obj_path": 车辆模型路径（str 类型，仅当 self.obj_path 非 None 时存在）；
                - "semantic_label": 语义标签（int 类型，仅当 self.semantic_label 非 None 时存在）。
        """
        # 基础属性字典（必选字段）
        vehicle = {"obj_index": int(self.obj_index), "obj_name": str(self.obj_name),
                   "location": [float(d) for d in self.location],
                   "rotation": float(self.rotation),
                   "scale": float(self.scale)}
        if self.obj_path is not None:
            vehicle["obj_path"] = self.obj_path
        if self.semantic_label is not None:
            vehicle["semantic_label"] = self.semantic_label
        return vehicle


scene_info_dict = {
    "dataset": "",
    "task": "",
    "weather": "sunny",
    "sequence": "",
    "bg_index": "",
    "vehicles": "",
    "nums_vehicles": ""
}


def generate_scene_info(dataset, sequence, bg_index, vehicles, assets):
    scene_info = copy.deepcopy(scene_info_dict)
    scene_info["dataset"] = dataset
    scene_info["sequence"] = sequence
    scene_info["bg_index"] = bg_index
    scene_info["vehicles"] = vehicles
    scene_info["assets"] = assets
    return scene_info


class SceneInfo(object):
    """
    场景信息封装类，用于管理场景的核心配置（数据集、天气、背景索引等）和车辆实例列表，
    支持从 JSON 文件/字典加载场景信息，以及将场景信息序列化为 JSON 格式（含/不含车辆信息），
    为场景生成、数据存储、流程复用等提供标准化的场景数据接口。

    核心功能：
    1. 支持多源输入（JSON 文件路径/字典），自动解析场景配置；
    2. 封装场景基础配置和车辆实例，提供车辆列表访问接口；
    3. 支持两种 JSON 序列化模式（含/不含车辆信息），适配不同存储/传输需求；
    4. 自动解析车辆信息并实例化为 Vehicle 对象，简化后续场景构建流程。
    """
    def __init__(self, source_info):
        """
        初始化场景信息对象，从 JSON 文件路径或字典中加载并解析场景配置。

        Args:
            source_info (Union[str, Dict]): 场景信息源，支持两种格式：
                - 字符串：JSON 文件路径（绝对路径或相对路径），文件需包含场景核心配置字段；
                - 字典：场景信息字典，需包含场景核心配置字段（dataset、weather、bg_index、assets）。
        """
        source = None
        if isinstance(source_info, str):
            source = self.load_json(source_info)
        elif isinstance(source_info, dict):
            source = source_info
        self.dataset = source["dataset"]
        self.weather = source["weather"]
        self.bg_index = source["bg_index"]
        self.assets = source["assets"]
        self.nums_vehicles = 0
        self.vehicles = []
        self.sequence = None
        if "vehicles" in source:
            vehicles = source["vehicles"]
            self.nums_vehicles = len(vehicles)
            self.vehicles = [Vehicle(info) for info in source["vehicles"]]

        if "sequence" in source:
            self.sequence = source["sequence"]

    def get_vehicles(self) -> List[Vehicle]:
        """
        获取场景中的所有车辆实例列表。

        Returns:
            List[Vehicle]: 车辆实例列表，无车辆时返回空列表。
        """
        return self.vehicles

    def to_json_without_vehicles(self, save_path=None):
        """
        将场景信息序列化为 JSON 兼容字典（不含车辆信息），支持保存到文件。

        适用于仅需存储场景基础配置的场景，减少存储开销。

        Args:
            save_path (Optional[str], 可选): JSON 文件保存路径，None 时仅返回字典不保存。

        Returns:
            Dict: 不含车辆信息的场景配置字典，包含 dataset、weather、bg_index、assets、nums_vehicles 字段，
                若 sequence 存在则包含该字段。
        """
        scene = {"dataset": self.dataset, "weather": self.weather, "bg_index": self.bg_index, "assets": self.assets,
                 "nums_vehicles": self.nums_vehicles}
        if self.sequence is not None:
            scene["sequence"] = self.sequence
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(scene, f)
        return scene

    def to_json_with_vehicles(self, save_path=None, indent=4):
        """
        将场景信息序列化为 JSON 兼容字典（含车辆信息），支持保存到文件。

        适用于完整场景信息存储，便于场景复现。

        Args:
            save_path (Optional[str], 可选): JSON 文件保存路径，None 时仅返回字典不保存；
            indent (int, 可选): JSON 格式化缩进字符数，默认 4，提升文件可读性。

        Returns:
            Dict: 含车辆信息的完整场景配置字典，在 to_json_without_vehicles 基础上增加 vehicles 字段，
                其值为车辆实例序列化后的字典列表。
        """
        scene = self.to_json_without_vehicles()
        vehicles = [vehicle.to_json() for vehicle in self.vehicles]
        scene["vehicles"] = vehicles
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(scene, f, indent=indent)
        return scene

    def load_json(self, json_path):
        """
        从 JSON 文件加载场景信息，返回解析后的字典。

        Args:
            json_path (str): JSON 文件路径（绝对/相对路径）。

        Returns:
            Dict: 解析后的场景信息字典。

        Raises:
            FileNotFoundError: 若文件不存在；
            json.JSONDecodeError: 若文件格式非法；
            UnicodeDecodeError: 若文件编码非 UTF-8。
        """
        with open(json_path, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        return json_file
