#!/usr/bin/env python3
"""
DRIVENCE: SemanticKITTI object insertion and point cloud generation tool.

Usage:
    python main.py point_cloud --bg_index=10 --obj_name=Car_1 --count=3
    python main.py scene --logic_scene_path=/path/to/scene.json
    python main.py group --bg_index=10
"""
import os
from pathlib import Path

import fire

# Ensure packaged modules know the project root directory. This must be set
# before importing modules that rely on drivence.utils.path_utils.
os.environ.setdefault("DRIVENCE_ROOT_DIR", str(Path(__file__).resolve().parent))

from entry.generate_data import (
    generate_data_with_point_cloud,
    generate_data_with_logic_scene,
    generate_data_with_group,
)


def point_cloud(bg_index=0, obj_name="Car_1", count=1):
    """
    Insert one or more instances of an object into a SemanticKITTI background frame and generate point cloud data.
    
    Args:
        bg_index: Background frame index (default: 0)
        obj_name: Object name, e.g., "Car_1", "Bus_935", "person_1" (default: "Car_1")
        count: Number of instances to insert (default: 1)
    """
    print(f"Inserting {count} instance(s) of {obj_name} into background frame {bg_index}")
    generate_data_with_point_cloud(bg_index, obj_name, count)
    print(f"Completed: {count} instance(s) of {obj_name} inserted into background frame {bg_index}")


def scene(logic_scene_path: str):
    """
    Generate data from a logic scene JSON file.
    
    Args:
        logic_scene_path: Path to the logic scene JSON file
    """
    print(f"Generating data from logic scene file: {logic_scene_path}")
    generate_data_with_logic_scene(logic_scene_path)
    print(f"Completed: data generated from logic scene file: {logic_scene_path}")


def group(bg_index: int):
    """
    Insert objects based on group definitions from obj_insert_config.yml.
    
    Args:
        bg_index: Background frame index
    """
    print(f"Starting group insertion, background frame: {bg_index}")
    generate_data_with_group(bg_index)
    print(f"Completed: group insertion finished, background frame: {bg_index}")


if __name__ == '__main__':
    fire.Fire({
        'point_cloud': point_cloud,
        'scene': scene,
        'group': group,
    })
