#!/usr/bin/env python3
"""
迭代式插入脚本
自动完成 insert_1 → insert_2 → ... → insert_5 的完整流程
无需手动修改 yml 配置文件
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _locate_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current] + list(current.parents):
        if (candidate / "entry" / "containers.py").exists():
            return candidate
    raise RuntimeError("Unable to locate Selda project root (missing entry/containers.py)")


PROJECT_ROOT = _locate_project_root()
os.environ.setdefault("DRIVENCE_ROOT_DIR", str(PROJECT_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_OUTPUT = os.environ.get("SELDA_RQ2_BASE_OUTPUT", "_outputs/rq2/0")
DEFAULT_BG_FILE = os.environ.get("SELDA_RQ2_BGS_FILE", "bgs_rq2_0.txt")


def backup_yml(yml_path: str) -> str:
    """
    备份 yml 文件
    
    Args:
        yml_path: yml 文件路径
        
    Returns:
        备份文件路径
    """
    backup_path = yml_path + ".backup"
    shutil.copy2(yml_path, backup_path)
    print(f"✓ 备份配置文件: {backup_path}")
    return backup_path


def restore_yml(yml_path: str, backup_path: str):
    """
    恢复 yml 文件
    
    Args:
        yml_path: yml 文件路径
        backup_path: 备份文件路径
    """
    shutil.copy2(backup_path, yml_path)
    print(f"✓ 恢复配置文件: {yml_path}")
    
    # 删除备份文件
    if os.path.exists(backup_path):
        os.remove(backup_path)
        print(f"✓ 删除备份文件: {backup_path}")


def modify_yml_for_iteration(
    yml_path: Path,
    iteration: int,
    base_output_dir: str,
    initial_demo_root: str,
):
    """
    修改 yml 文件以适配当前迭代
    
    使用文本替换方式，保留原始 yml 格式和注释
    
    Args:
        yml_path: yml 文件路径
        iteration: 当前迭代次数（1-5）
        base_output_dir: 输出基础目录（如 "_outputs/rq2/0"）
        initial_demo_root: 原始 demo_dataset.root_dir 值
    """
    yml_path = Path(yml_path)
    base_output_dir = base_output_dir.rstrip("/")
    
    # 第一轮：从原始数据集开始
    if iteration == 1:
        demo_root = initial_demo_root
    else:
        # 后续轮次：从上一轮的输出开始
        demo_root = f"{base_output_dir}/insert_{iteration - 1}"
    
    # 当前轮次的输出目录
    aug_root = f"{base_output_dir}/insert_{iteration}"
    
    # 读取原始文件内容
    with open(yml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 逐行查找并替换
    modified_lines = []
    in_aug_dataset = False
    in_demo_dataset = False
    
    for line in lines:
        # 检测进入 aug_dataset 或 demo_dataset 区块
        if line.strip().startswith('aug_dataset:'):
            in_aug_dataset = True
            in_demo_dataset = False
            modified_lines.append(line)
        elif line.strip().startswith('demo_dataset:'):
            in_demo_dataset = True
            in_aug_dataset = False
            modified_lines.append(line)
        elif line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            # 遇到新的顶级键，退出当前区块
            in_aug_dataset = False
            in_demo_dataset = False
            modified_lines.append(line)
        else:
            # 在区块内，查找并替换 root_dir
            if in_aug_dataset and 'root_dir:' in line:
                indent = len(line) - len(line.lstrip())
                modified_lines.append(' ' * indent + f'root_dir: "{aug_root}"\n')
            elif in_demo_dataset and 'root_dir:' in line:
                indent = len(line) - len(line.lstrip())
                modified_lines.append(' ' * indent + f'root_dir: "{demo_root}"\n')
            else:
                modified_lines.append(line)
    
    # 写回文件
    with open(yml_path, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)
    
    print(f"✓ 修改配置:")
    print(f"    demo_dataset.root_dir: {demo_root}")
    print(f"    aug_dataset.root_dir:  {aug_root}")


def run_rq2(working_dir: Path, bgs_file: Path, base_output_dir: str) -> int:
    """
    运行 RQ2.py
    
    Args:
        working_dir: 工作目录（Selda 项目根目录）
        bgs_file: 背景索引文件路径
        base_output_dir: 输出基础目录（相对路径字符串）
    """
    print(f"\n{'='*60}")
    print("开始运行 RQ2.py...")
    print(f"{'='*60}\n")
    
    env = os.environ.copy()
    env["SELDA_RQ2_BGS_FILE"] = str(bgs_file)
    env["SELDA_RQ2_BASE_OUTPUT"] = base_output_dir
    
    # 运行 RQ2
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "RQ2.py")],
        cwd=str(working_dir),
        env=env,
        check=False,
    )
    
    if result.returncode != 0:
        print(f"\n⚠️  RQ2.py 运行结束，返回码: {result.returncode}")
    else:
        print(f"\n✓ RQ2.py 运行完成")
    
    return result.returncode


def main():
    """主函数：执行5轮迭代式插入"""
    
    print("="*60)
    print("迭代式插入脚本")
    print("="*60)
    print("功能: 自动完成 insert_1 → insert_2 → ... → insert_5")
    print("="*60 + "\n")
    
    # 配置路径
    working_dir = PROJECT_ROOT
    yml_path = PROJECT_ROOT / "configs" / "dataset" / "semantic_kitti.yml"
    base_output_dir = DEFAULT_BASE_OUTPUT.rstrip("/")
    
    bgs_file = Path(DEFAULT_BG_FILE)
    if not bgs_file.is_absolute():
        bgs_file = TOOLS_DIR / bgs_file
    
    # 检查配置文件是否存在
    if not yml_path.exists():
        print(f"❌ 错误: 找不到配置文件: {yml_path}")
        return
    if not bgs_file.exists():
        print(f"⚠️  提示: 背景索引文件不存在，运行时将由 RQ2.py 自动生成: {bgs_file}")
    
    with open(yml_path, 'r', encoding='utf-8') as f:
        dataset_cfg = yaml.safe_load(f)
    initial_demo_root = dataset_cfg.get("demo_dataset", {}).get("root_dir", "_data/semanticKITTI/dataset")
    
    print(f"配置文件: {yml_path}")
    print(f"输出目录: {base_output_dir}")
    print(f"工作目录: {working_dir}")
    print(f"背景索引文件: {bgs_file}\n")
    
    # 备份原始配置
    backup_path = backup_yml(str(yml_path))
    
    try:
        # 执行 5 轮迭代
        for iteration in range(1, 6):
            print(f"\n{'#'*60}")
            print(f"# 第 {iteration} 轮迭代: insert_{iteration}")
            print(f"{'#'*60}\n")
            
            # 修改配置文件
            modify_yml_for_iteration(yml_path, iteration, base_output_dir, initial_demo_root)
            
            # 运行 RQ2
            return_code = run_rq2(working_dir, bgs_file, base_output_dir)
            
            if return_code != 0:
                print(f"\n⚠️  第 {iteration} 轮出现错误，但继续执行下一轮...")
            
            print(f"\n✓ 第 {iteration} 轮完成")
            print(f"{'#'*60}\n")
        
        print(f"\n{'='*60}")
        print("✅ 所有 5 轮迭代完成！")
        print(f"{'='*60}")
        print(f"结果保存在:")
        for i in range(1, 6):
            output_dir = f"{base_output_dir}/insert_{i}"
            print(f"  第 {i} 轮: {output_dir}")
        print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断，正在恢复配置文件...")
    
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 无论如何都要恢复原始配置
        print(f"\n正在恢复原始配置...")
        restore_yml(str(yml_path), backup_path)
        print(f"✓ 配置文件已恢复\n")


if __name__ == '__main__':
    main()

