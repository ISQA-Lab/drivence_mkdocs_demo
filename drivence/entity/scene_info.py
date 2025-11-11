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
    def __init__(self, info: dict):
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

    def to_dict(self):
        return self.to_json()

    def to_json(self):
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
    def __init__(self, source_info):
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
        return self.vehicles

    def to_json_without_vehicles(self, save_path=None):
        """
        将scene_info转化为json——不包含vehicles
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
        将scene_info转化为json——包含vehicles
        """
        scene = self.to_json_without_vehicles()
        vehicles = [vehicle.to_json() for vehicle in self.vehicles]
        scene["vehicles"] = vehicles
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(scene, f, indent=indent)
        return scene

    def load_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        return json_file
