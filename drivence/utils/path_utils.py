import os
import yaml
import sys
import pathlib
from pathlib import Path
# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
# run->edit configuration->Working Directory-> set to project root xxx/Drivence


def symlink(input_path, output_path):
    if not os.path.exists(input_path):
        raise ValueError("input: ", input_path)
    if os.path.exists(output_path):
        os.remove(output_path)  # can delet soft link files
    os.symlink(input_path, output_path)
    # print(input_path, "-->", output_path)

def _find_project_structure() -> Path:
    current = Path(__file__).resolve()
    for ancestor in current.parents:
        candidate = ancestor / "configs" / "project_structure.yml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Unable to locate 'configs/project_structure.yml'. "
        "Ensure the configuration directory is available next to the project."
    )


def get_project_root_dir():
    env_root = os.environ.get("DRIVENCE_ROOT_DIR")
    if env_root:
        return env_root

    config_path = _find_project_structure()

    with config_path.open("r", encoding="utf-8") as f:
        project_yaml = yaml.load(f, Loader=yaml.FullLoader)

    root_dir = project_yaml['project_structure']['root_dir']
    if not os.path.isabs(root_dir):
        root_dir = str((config_path.parent / root_dir).resolve())
    return root_dir


def create_parent_dir(path,exist_ok=True):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir,exist_ok=exist_ok)


def load_files_in_dir(dir_path,suffix=None):
    from natsort import natsorted
    files = os.listdir(dir_path)
    if suffix is not None:
        files = [file for file in files if file.endswith(suffix)]
    natsorted_files = natsorted(files)
    return natsorted_files


if __name__ == '__main__':
    print(get_project_root_dir())