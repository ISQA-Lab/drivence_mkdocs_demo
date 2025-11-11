# Selda

Selda lets you inject custom objects into SemanticKITTI-style scenes and regenerate labelled LiDAR point clouds. The repo ships with turn-key augmentation scripts plus optional C/Cython builds for performance.

## Requirements

- Linux (Ubuntu 20.04+ recommended)
- Python 3.9+
- NVIDIA GPU with CUDA/cuDNN (for acceleration)
- PyTorch, Open3D, OpenCV (installed via `requirements.txt`)

## Setup

Prepare the Python environment and materialise the precompiled extensions once:

```bash
pip install -r requirements.txt
python build_script.py   --artifact-dir lidar_core   --dist-dir lidar_core_dist   --name lidar_core   --version 1.0.0
pip install lidar_core_dist/lidar_core-1.0.0-py3-none-any.whl
```

The wheel command copies the bundled `.so/.pyi` assets into a temporary staging area, writes a minimal `setup.py`, and drops the resulting wheel into `dist/`. Install the produced wheel if you want to import the packaged modules elsewhere.

## Generate Augmented Data

```bash
# Insert three copies of Car_1 into frame 1
python main.py point_cloud --bg_index 1 --obj_name Car_1 --count 3

# Insert the object groups defined in configs/obj_insert_config.yml
python main.py group --bg_index 0

# Insert the objects defined in a logic-scene JSON
python main.py scene --logic_scene_path scene_example/car_example.json
```

All generated point clouds, labels, and logs are written under `_outputs/` (unless you overrode the paths in `configs/`).