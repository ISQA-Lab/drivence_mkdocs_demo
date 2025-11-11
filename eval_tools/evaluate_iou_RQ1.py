#!/usr/bin/env python3
# Evaluation script for RQ1 experiments (Re1, Re2, Re3, Re4, etc.)
# Evaluates each category and compares with baseline 08_

import argparse
import os
import re
import yaml
import sys
import subprocess
import numpy as np
import torch
import pandas as pd

from modules.ioueval import iouEval
from common.laserscan import SemLaserScan

def eval_category(pred_path, label_path, scan_path, eval_mode, target_class=None, accept_classes=None):
    """
    Evaluate a specific category with custom evaluation mode
    
    Args:
        pred_path: Path to predictions directory
        label_path: Path to labels directory
        scan_path: Path to scans directory
        eval_mode: 'standard', 'bicycle_relaxed', 'inserted_relaxed_dynamic', 'inserted_relaxed_infrastructure'
        target_class: Target class to get IoU for (learning map class)
        accept_classes: List of acceptable classes for relaxed evaluation
    
    Returns:
        Dictionary with iou scores
    """
    # Get all label and prediction files
    matched = []
    
    if os.path.isdir(pred_path):
        pred_files = [f for f in os.listdir(pred_path) if f.endswith('.label')]
        pred_files.sort()
        
        for pred_name in pred_files:
            label_name = pred_name  # Same name
            pred_file = os.path.join(pred_path, pred_name)
            label_file = os.path.join(label_path, label_name)
            scan_name = pred_name.replace('.label', '.bin')
            scan_file = os.path.join(scan_path, scan_name)
            
            if os.path.exists(label_file) and os.path.exists(scan_file):
                matched.append((scan_file, label_file, pred_file))
    
    # create evaluator
    nr_classes = len(class_inv_remap)
    device = torch.device("cpu")
    
    # 动态决定是否忽略类别20
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
    
    # 根据eval_mode决定是否忽略类别20（inserted_object）
    # 只有特定的relaxed模式才评估类别20
    if eval_mode not in ['inserted_relaxed_dynamic', 'inserted_relaxed_infrastructure']:
        if 20 not in ignore:
            ignore.append(20)
    
    evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()
    
    # Evaluate each scan
    print(f"  Matched {len(matched)} files for evaluation")
    if len(matched) == 0:
        print(f"  WARNING: No files matched! pred_path={pred_path}, label_path={label_path}")
    
    for scan_file, label_file, pred_file in matched:
        # open label
        label = SemLaserScan(project=False)
        label.open_scan(scan_file)
        label.open_label(label_file)
        u_label_sem = remap_lut[label.sem_label]
        
        # open prediction
        pred = SemLaserScan(project=False)
        pred.open_scan(scan_file)
        pred.open_label(pred_file)
        u_pred_sem = remap_lut[pred.sem_label]
        
        # Apply special evaluation modes
        if eval_mode == 'bicycle_relaxed':
            # For bicycle, accept motorcycle predictions as correct
            bicycle_class = 2
            motorcycle_class = 3
            bicycle_gt_mask = (u_label_sem == bicycle_class)
            relaxed_pred = u_pred_sem.copy()
            relaxed_pred[(bicycle_gt_mask) & (u_pred_sem == motorcycle_class)] = bicycle_class
            u_pred_sem = relaxed_pred
        
        elif eval_mode == 'inserted_relaxed_dynamic':
            # For inserted objects (label 98), accept dynamic foreground classes
            inserted_object_class = 20
            inserted_gt_mask = (u_label_sem == inserted_object_class)
            relaxed_pred = u_pred_sem.copy()
            dynamic_foreground_classes = [1, 2, 3, 4, 5, 6, 7, 8]  # vehicles and humans
            for dyn_class in dynamic_foreground_classes:
                relaxed_pred[(inserted_gt_mask) & (u_pred_sem == dyn_class)] = inserted_object_class
            u_pred_sem = relaxed_pred
        
        elif eval_mode == 'inserted_relaxed_infrastructure':
            # For inserted objects (label 98), accept infrastructure classes
            inserted_object_class = 20
            inserted_gt_mask = (u_label_sem == inserted_object_class)
            relaxed_pred = u_pred_sem.copy()
            infrastructure_classes = [14, 18, 19]  # fence, pole, traffic-sign
            for inf_class in infrastructure_classes:
                relaxed_pred[(inserted_gt_mask) & (u_pred_sem == inf_class)] = inserted_object_class
            u_pred_sem = relaxed_pred
        
        elif eval_mode == 'motorcyclist_relaxed':
            # For motorcyclist ground truth (class 8), accept person (6) or motorcycle (3) predictions
            motorcyclist_class = 8
            person_class = 6
            motorcycle_class = 3
            motorcyclist_gt_mask = (u_label_sem == motorcyclist_class)
            relaxed_pred = u_pred_sem.copy()
            relaxed_pred[(motorcyclist_gt_mask) & (u_pred_sem == person_class)] = motorcyclist_class
            relaxed_pred[(motorcyclist_gt_mask) & (u_pred_sem == motorcycle_class)] = motorcyclist_class
            u_pred_sem = relaxed_pred
        
        elif eval_mode == 'accept_specific':
            # Accept specific classes
            if accept_classes is not None:
                target_mask = (u_label_sem == target_class)
                relaxed_pred = u_pred_sem.copy()
                for accept_class in accept_classes:
                    relaxed_pred[(target_mask) & (u_pred_sem == accept_class)] = target_class
                u_pred_sem = relaxed_pred
        
        evaluator.addBatch(u_pred_sem, u_label_sem)
    
    # Get results
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()
    
    print(f"  Evaluated, mIoU: {m_jaccard.item():.3f}, Accuracy: {m_accuracy.item():.3f}")
    
    results = {
        'accuracy': m_accuracy.item() if hasattr(m_accuracy, 'item') else m_accuracy,
        'mean_iou': m_jaccard.item() if hasattr(m_jaccard, 'item') else m_jaccard,
        'class_iou': [j.item() if hasattr(j, 'item') else j for j in class_jaccard]
    }
    
    # Get specific class IoU if requested
    if target_class is not None:
        if target_class < len(class_jaccard):
            results['target_iou'] = class_jaccard[target_class].item() if hasattr(class_jaccard[target_class], 'item') else class_jaccard[target_class]
        else:
            results['target_iou'] = 0.0
    
    return results


# Parse arguments
parser = argparse.ArgumentParser("./evaluate_rq1_re1.py")
parser.add_argument('--dataset', '-d', type=str, default="/home/atri/WD_Passport/semanticKITTI/dataset/", help='Dataset dir.')
parser.add_argument('--predictions', '-p', type=str, default="./RQ1_Re5/salsa", help='Predictions dir (e.g., ./RQ1_Re1/lsk, ./RQ1_Re2/cenet).')
parser.add_argument('--data_cfg', '-dc', type=str, default="config/labels/semantic-kitti.yaml", help='Dataset config file.')

FLAGS = parser.parse_args()

# Load config
DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
class_strings = DATA["labels"]
class_remap = DATA["learning_map"]
class_inv_remap = DATA["learning_map_inv"]
class_ignore = DATA["learning_ignore"]

    # Add custom mapping for label 98
class_remap[98] = 20
class_inv_remap[20] = 98
class_ignore_original = class_ignore.copy()  # 保存原始忽略设置
class_strings[98] = "inserted-object"
nr_classes = len(class_inv_remap)

# Make lookup table
maxkey = 0
for key, data in class_remap.items():
    if key > maxkey:
        maxkey = key
remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
for key, data in class_remap.items():
    try:
        remap_lut[key] = data
    except IndexError:
        print("Wrong key ", key)

# Define evaluation mappings based on user's requirements
# Mapping: (sequence_name, eval_mode, target_classes, accept_classes for averaging)
EVAL_RULES = {
    '08_car': ('standard', 1, None),  # car class
    '08_bus': ('standard', 5, None),  # other_vehicle class
    '08_truck': ('standard', 4, None),  # truck class
    '08_motorcycle': ('standard', 3, None),  # motorcycle class
    '08_bicycle': ('bicycle_relaxed', 2, None),  # bicycle with relaxed mode
    '08_tractor': ('standard', 5, None),  # other_vehicle
    '08_scooter': ('standard', 5, None),  # other_vehicle
    '08_ambulance': ('standard', 5, None),  # other_vehicle
    
    # Human categories -> person (class 6)
    '08_adult': ('standard', 6, None),
    '08_senior': ('standard', 6, None),
    '08_police': ('standard', 6, None),
    '08_worker': ('standard', 6, None),
    '08_kid': ('standard', 6, None),
    
    # Infrastructure
    '08_fence': ('standard', 14, None),  # fence
    '08_street_light': ('standard', 18, None),  # pole
    '08_traffic_sign': ('standard', [18, 19], None),  # pole and traffic-sign average
    '08_trafficcone': ('inserted_relaxed_infrastructure', None, None),  # use overall miou
    '08_telephone_pole': ('standard', 18, None),  # pole
    
    # Vegetation
    '08_shrubbery': ('standard', 15, None),  # vegetation
    '08_tree': ('standard', [15, 16], None),  # vegetation and trunk average
    
    # Animals and others -> inserted object with dynamic foreground mode, use overall miou
    '08_dog': ('inserted_relaxed_dynamic', None, None),  # use overall miou
    '08_cat': ('inserted_relaxed_dynamic', None, None),
    '08_boar': ('inserted_relaxed_dynamic', None, None),
    '08_luggage': ('inserted_relaxed_dynamic', None, None),
    '08_stroller': ('inserted_relaxed_dynamic', None, None),
    '08_wheelchair': ('inserted_relaxed_dynamic', None, None),
    
    # Cyclist
    '08_person-riding-bicycle': ('standard', 7, None),  # bicyclist
    '08_person-riding-motor': ('motorcyclist_relaxed', 8, None),  # motorcyclist relaxed mode
    '08_person-pushing-bicycle': ('standard', 7, None),  # bicyclist
    '08_person-pushing-motorcycle': ('motorcyclist_relaxed', 8, None),  # motorcyclist relaxed mode
    
    # Pedestrian -> special handling (person+inserted_object for cur, 8-classes average for ori)
    '08_person-pulling-luggage': ('inserted_relaxed_dynamic', None, None),  # dynamic foreground mode for luggage
    '08_dog-walker': ('inserted_relaxed_dynamic', None, None),  # will be handled specially
}

# Get baseline 08_ results first
print("Evaluating baseline 08_...")
baseline_pred_path = os.path.join(FLAGS.predictions, "08_", "sequences", "08", "predictions")
# Note: predictions use sequence "08" name, dataset uses sequence "08_"
baseline_label_path = os.path.join(FLAGS.dataset, "sequences", "08_", "labels")
baseline_scan_path = os.path.join(FLAGS.dataset, "sequences", "08_", "velodyne")

baseline_results = {}
# Baseline使用standard模式，自动忽略类别20（不包含98号物体的baseline不应该考虑类别20）
print(f"Baseline pred path: {baseline_pred_path}")
print(f"Baseline label path: {baseline_label_path}")
print(f"Baseline scan path: {baseline_scan_path}")
baseline_overall = eval_category(baseline_pred_path, baseline_label_path, baseline_scan_path, 'standard')
baseline_results['overall'] = baseline_overall['mean_iou']
baseline_results['classes'] = baseline_overall['class_iou']

# 对于pedestrian类别需要用到类别20的baseline，我们需要单独计算
# 但对于baseline本身（原始序列08），不包含98号物体，所以类别20的IoU应该设为0
baseline_results['class_20_iou'] = 0.0  # Baseline中不包含98号物体

# Convert IoU to percentages (multiply by 100)
baseline_results['overall'] = baseline_results['overall'] * 100
baseline_results['classes'] = [c * 100 for c in baseline_results['classes']]

print(f"\nBaseline overall mIoU: {baseline_results['overall']:.3f}")
print(f"Baseline car (class 1) IoU: {baseline_results['classes'][1]:.3f}")
print(f"Baseline person (class 6) IoU: {baseline_results['classes'][6]:.3f}")

# Evaluate all other sequences
results_table = []

sequences = [d for d in os.listdir(FLAGS.predictions) if d.startswith('08_') and os.path.isdir(os.path.join(FLAGS.predictions, d))]

for seq_name in sorted(sequences):
    if seq_name == '08_':
        continue
    
    print(f"\nEvaluating {seq_name}...")
    
    # Get paths
    # Predictions are stored under sequences/08/ directory
    pred_path = os.path.join(FLAGS.predictions, seq_name, "sequences", "08", "predictions")
    # Labels are in the modified sequence directory
    label_path = os.path.join(FLAGS.dataset, "sequences", seq_name, "labels")
    # Scans should use modified sequence data (velodyne files have same names)
    scan_path = os.path.join(FLAGS.dataset, "sequences", seq_name, "velodyne")
    
    # Check if all required directories exist
    if not os.path.exists(pred_path):
        print(f"  Skipping {seq_name}: predictions directory not found")
        continue
    if not os.path.exists(label_path):
        print(f"  Skipping {seq_name}: labels directory not found")
        continue
    if not os.path.exists(scan_path):
        print(f"  Skipping {seq_name}: scans directory not found")
        continue
    
    if seq_name not in EVAL_RULES:
        print(f"Warning: No evaluation rule for {seq_name}")
        continue
    
    eval_mode, target_classes, accept_classes = EVAL_RULES[seq_name]
    
    # Evaluate this sequence
    seq_results = eval_category(pred_path, label_path, scan_path, eval_mode)
    
    # Get IoU based on rules (convert to percentages)
    if isinstance(target_classes, list):
        # Average of multiple classes
        ious = [seq_results['class_iou'][tc] for tc in target_classes if tc < len(seq_results['class_iou'])]
        iou_cur = np.mean(ious) * 100 if ious else 0.0
        
        # Baseline IoU (already in percentages)
        baseline_ious = [baseline_results['classes'][tc] for tc in target_classes if tc < len(baseline_results['classes'])]
        iou_ori = np.mean(baseline_ious) if baseline_ious else 0.0
    elif target_classes is None:
        # Use overall mean IoU for animal/others/trafficcone categories
        iou_cur = seq_results['mean_iou'] * 100
        iou_ori = baseline_results['overall']
    else:
        # Single class
        iou_cur = (seq_results['class_iou'][target_classes] if target_classes < len(seq_results['class_iou']) else 0.0) * 100
        iou_ori = baseline_results['classes'][target_classes] if target_classes < len(baseline_results['classes']) else 0.0
    
    # Special handling for motor rider/pusher baseline comparison
    if seq_name in ['08_person-riding-motor', '08_person-pushing-motorcycle']:
        person_class = 6
        motorcycle_class = 3
        baseline_pair = []
        if person_class < len(baseline_results['classes']):
            baseline_pair.append(baseline_results['classes'][person_class])
        if motorcycle_class < len(baseline_results['classes']):
            baseline_pair.append(baseline_results['classes'][motorcycle_class])
        iou_ori = np.mean(baseline_pair) if baseline_pair else 0.0

    # Special handling for pedestrian categories
    if seq_name in ['08_person-pulling-luggage', '08_dog-walker']:
        # Pedestrian categories: 
        # iou_cur = average of person (6) and inserted_object (20) with dynamic relaxed mode
        # iou_ori = average of car,truck,motorcycle,bicycle,other-vehicle,person,bicyclist,motorcyclist
        person_class = 6
        inserted_class = 20
        
        # iou_cur: person + inserted_object average (already evaluated with inserted_relaxed_dynamic mode)
        if person_class < len(seq_results['class_iou']) and inserted_class < len(seq_results['class_iou']):
            iou_cur = (seq_results['class_iou'][person_class] + seq_results['class_iou'][inserted_class]) / 2.0 * 100
        else:
            iou_cur = (seq_results['class_iou'][person_class] if person_class < len(seq_results['class_iou']) else 0.0) * 100
        
        # iou_ori: 8-classes average for baseline
        foreground_classes = [1, 4, 3, 2, 5, 6, 7, 8]  # car,truck,motorcycle,bicycle,other-vehicle,person,bicyclist,motorcyclist
        baseline_foreground_ious = [baseline_results['classes'][c] for c in foreground_classes if c < len(baseline_results['classes'])]
        iou_ori = np.mean(baseline_foreground_ious) if baseline_foreground_ious else 0.0
    
    iou_diff = iou_ori - iou_cur
    
    results_table.append({
        'sequence': seq_name,
        'iou_cur': iou_cur,
        'iou_ori': iou_ori,
        'iou_diff': iou_diff
    })
    
    print(f"  IoU current: {iou_cur:.3f}, IoU original: {iou_ori:.3f}, Diff: {iou_diff:.3f}")

# Create output table matching user's format
print("\n" + "="*80)
print("RESULTS TABLE")
print("="*80)

# Define parent categories and their children
PARENT_CATEGORIES = {
    'Vehicle': ['08_car', '08_bus', '08_truck', '08_motorcycle', '08_bicycle', '08_tractor', '08_scooter', '08_ambulance'],
    'Human': ['08_adult', '08_senior', '08_police', '08_worker', '08_kid'],
    'Infrastructure': ['08_fence', '08_street_light', '08_traffic_sign', '08_telephone_pole', '08_trafficcone'],
    'Vegetation': ['08_shrubbery', '08_tree'],
    'Animal': ['08_dog', '08_cat', '08_boar'],
    'Others': ['08_luggage', '08_stroller', '08_wheelchair'],
    'Cyclist': ['08_person-riding-bicycle', '08_person-riding-motor', '08_person-pushing-bicycle', '08_person-pushing-motorcycle'],
    'Pedestrian': ['08_person-pulling-luggage', '08_dog-walker']
}

# Map sequence names to display names
DISPLAY_NAMES = {
    '08_car': 'Car', '08_bus': 'Bus', '08_truck': 'Truck', '08_motorcycle': 'Motorcycle',
    '08_bicycle': 'Bicycle', '08_tractor': 'Tractor', '08_scooter': 'Scooter', '08_ambulance': 'Ambulance',
    '08_adult': 'Adult', '08_senior': 'Senior', '08_police': 'Police', '08_worker': 'Worker', '08_kid': 'Kid',
    '08_fence': 'Fence', '08_street_light': 'Street_light', '08_traffic_sign': 'Traffic_sign',
    '08_trafficcone': 'Trafficcone', '08_telephone_pole': 'Telephone_pole',
    '08_shrubbery': 'Shrubbery', '08_tree': 'Tree',
    '08_dog': 'Dog', '08_cat': 'Cat', '08_boar': 'Boar',
    '08_luggage': 'Luggage', '08_stroller': 'Stroller', '08_wheelchair': 'Wheelchair',
    '08_person-riding-bicycle': 'Bike_rider', '08_person-riding-motor': 'Motor_rider',
    '08_person-pushing-bicycle': 'Bike_pusher', '08_person-pushing-motorcycle': 'Motor_pusher',
    '08_person-pulling-luggage': 'Luggage_puller', '08_dog-walker': 'Dog_walker'
}

# Build table
table_rows = []
for parent, children in PARENT_CATEGORIES.items():
    for child in children:
        # Find result
        result = next((r for r in results_table if r['sequence'] == child), None)
        if result:
            display_name = DISPLAY_NAMES.get(child, child)
            table_rows.append({
                'parent': parent,
                'subcategory': display_name,
                'iou_cur': result['iou_cur'],
                'iou_ori': result['iou_ori'],
                'iou_diff': result['iou_diff']
            })
        else:
            # Category not found, add empty row
            display_name = DISPLAY_NAMES.get(child, child)
            table_rows.append({
                'parent': parent,
                'subcategory': display_name,
                'iou_cur': None,
                'iou_ori': None,
                'iou_diff': None
            })

# Print table
print(f"{'Parent':<15} {'Subcategory':<20} {'iou_cur':<10} {'iou_ori':<10} {'iou_diff':<10}")
print("-" * 80)
for row in table_rows:
    if row['iou_cur'] is not None:
        print(f"{row['parent']:<15} {row['subcategory']:<20} {row['iou_cur']:<10.3f} {row['iou_ori']:<10.3f} {row['iou_diff']:<10.3f}")
    else:
        print(f"{row['parent']:<15} {row['subcategory']:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

# Extract experiment number from predictions path (e.g., "RQ1_Re1" -> "Re1")
# This allows the script to work with Re1, Re2, Re3, Re4, etc.
prediction_dir = os.path.basename(os.path.dirname(FLAGS.predictions))
match = re.search(r'(Re\d+)', prediction_dir)
exp_suffix = match.group(1) if match else "unknown"

# Save to xlsx
df = pd.DataFrame(table_rows)
output_file = os.path.join(FLAGS.predictions, f"rq1_{exp_suffix}_results.xlsx")
try:
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\nResults saved to {output_file}")
except ImportError:
    print("\nWarning: openpyxl not installed. Attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Results saved to {output_file}")

