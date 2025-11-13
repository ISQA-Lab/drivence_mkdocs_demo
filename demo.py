#!/usr/bin/env python3
"""
Simple demo script that inserts two `Car_1` instances into the background frame
with index 0 using the standard Selda data generation pipeline.

Run:
    python demo.py
"""

import os
import sys

from drivence.utils.path_utils import get_project_root_dir

PROJECT_ROOT = get_project_root_dir()

os.environ.setdefault("DRIVENCE_ROOT_DIR", PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entry.generate_data import generate_data_with_point_cloud


def main() -> None:
    bg_index = 0
    obj_name = "Car_1"
    count = 2

    print(f"Inserting {count} copies of {obj_name} into background frame {bg_index}...")
    generate_data_with_point_cloud(bg_index=bg_index, obj_name=obj_name, count=count)
    print("Done. Generated assets are available under the configured output directory.")


if __name__ == "__main__":
    main()
