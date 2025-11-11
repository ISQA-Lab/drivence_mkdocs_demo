#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Diversity Metrics Evaluation
# - Data Diversity: Compare original GT labels vs transformed GT labels distribution
# - Fault Diversity: Compare original FP distribution vs transformed FP distribution

import argparse
import os
import yaml
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import glob
import torch
from modules.ioueval import iouEval

def hellinger_distance(p, q):
    """
    Compute Hellinger distance between two probability distributions.
    H(P, Q) = sqrt(0.5 * sum((sqrt(p_i) - sqrt(q_i))^2))
    """
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

def get_label_distribution(label_files, remap_lut, nr_classes):
    """
    Get distribution of point categories from label files.
    
    Returns:
        prob_dist: Probability distribution over categories
        total_counts: Total point counts per category
    """
    total_counts = np.zeros(nr_classes, dtype=np.int64)
    
    for label_file in label_files:
        # Read label file
        label = np.fromfile(label_file, dtype=np.uint32).reshape((-1))
        sem_label = label & 0xFFFF
        sem_label_remapped = remap_lut[sem_label]
        
        # Compute histogram
        hist, _ = np.histogram(sem_label_remapped, bins=nr_classes, range=(0, nr_classes))
        total_counts += hist.astype(np.int64)
    
    # Normalize to probability distribution
    total_points = total_counts.sum()
    if total_points > 0:
        prob_dist = total_counts.astype(np.float64) / total_points
    else:
        prob_dist = np.zeros(nr_classes, dtype=np.float64)
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    prob_dist = prob_dist + epsilon
    prob_dist = prob_dist / prob_dist.sum()
    
    return prob_dist, total_counts

def get_fp_distribution(gt_files, pred_files, remap_lut, nr_classes, ignore_list, 
                        apply_relaxation=False, class_inv_remap=None):
    """
    Get distribution of False Positive errors.
    
    Returns:
        fp_dist: Probability distribution of FP over categories
        total_fp_counts: Total FP counts per category
    """
    # Create evaluator
    device = torch.device("cpu")
    evaluator = iouEval(nr_classes, device, ignore_list)
    evaluator.reset()
    
    # Process all files
    for gt_file, pred_file in zip(gt_files, pred_files):
        # Read labels
        label_gt = np.fromfile(gt_file, dtype=np.uint32).reshape((-1))
        sem_label_gt = remap_lut[label_gt & 0xFFFF]
        label_pred = np.fromfile(pred_file, dtype=np.uint32).reshape((-1))
        sem_label_pred = remap_lut[label_pred & 0xFFFF]
        
        # Apply relaxation for label 98 if needed
        if apply_relaxation:
            inserted_object_class = 20
            inserted_gt_mask = (sem_label_gt == inserted_object_class)
            sem_label_pred_relaxed = sem_label_pred.copy()
            dynamic_foreground_classes = [1, 2, 3, 4, 5, 6, 7, 8]
            for dyn_class in dynamic_foreground_classes:
                sem_label_pred_relaxed[inserted_gt_mask & (sem_label_pred == dyn_class)] = inserted_object_class
        else:
            sem_label_pred_relaxed = sem_label_pred
        
        # Add to evaluator
        evaluator.addBatch(sem_label_pred_relaxed, sem_label_gt)
    
    # Get FP counts
    tp, fp, fn = evaluator.getStats()
    total_fp_counts = fp.cpu().numpy().astype(np.int64)
    
    # Normalize to probability distribution
    total_fp = total_fp_counts.sum()
    if total_fp > 0:
        fp_dist = total_fp_counts.astype(np.float64) / total_fp
    else:
        fp_dist = np.zeros(nr_classes, dtype=np.float64)
    
    # Add epsilon
    epsilon = 1e-10
    fp_dist = fp_dist + epsilon
    fp_dist = fp_dist / fp_dist.sum()
    
    return fp_dist, total_fp_counts

def compute_diversity_metrics(dist1, dist2):
    """Compute KL, JS, WD, HD between two distributions."""
    kl = entropy(dist1, dist2)
    js_dist = jensenshannon(dist2, dist1)
    js = js_dist ** 2
    positions = np.arange(len(dist1))
    wd = wasserstein_distance(positions, positions, dist2, dist1)
    hd = hellinger_distance(dist2, dist1)
    return {'KL': kl, 'JS': js, 'WD': wd, 'HD': hd}

def save_to_log(logfile, message):
    """Save message to log file."""
    with open(logfile, 'a') as f:
        f.write(message + '\n')
    print(message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_diversity_metrics.py")
    parser.add_argument(
        '--original_gt', '-og',
        type=str,
        default="/home/atri/WD_Passport/semanticKITTI/dataset/sequences/08_i0",
        help='Original GT labels directory. Defaults to %(default)s',
    )
    parser.add_argument(
        '--original_pred', '-op',
        type=str,
        default="./RQ2/0/insert_0",
        help='Original predictions directory. Defaults to %(default)s',
    )
    parser.add_argument(
        '--transformed_gt', '-tg',
        type=str,
        default="/home/atri/WD_Passport/semanticKITTI/dataset/sequences/08_i3",
        help='Transformed GT labels directory. Defaults to %(default)s',
    )
    parser.add_argument(
        '--transformed_pred', '-tp',
        type=str,
        default="./RQ2/2/insert_3",
        help='Transformed predictions directory. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--include_inserted_object',
        action='store_true',
        default=True,
        help='Include label 98 (inserted object) in evaluation. Default: True',
    )
    parser.add_argument(
        '--output', '-out',
        type=str,
        default="diversity_metrics.txt",
        help='Output log file. Defaults to %(default)s',
    )

    FLAGS = parser.parse_args()

    print("*" * 80)
    print("DIVERSITY METRICS EVALUATION")
    print("Original GT: ", FLAGS.original_gt)
    print("Original Pred: ", FLAGS.original_pred)
    print("Transformed GT: ", FLAGS.transformed_gt)
    print("Transformed Pred: ", FLAGS.transformed_pred)
    print("Include inserted object: ", FLAGS.include_inserted_object)
    print("*" * 80)

    # Open data config file
    try:
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # Get label mappings
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA.get("learning_map_inv", {})
    class_strings = DATA.get("labels", {})
    class_ignore_dict = DATA["learning_ignore"].copy()
    
    # Handle label 98
    if FLAGS.include_inserted_object:
        class_remap[98] = 20
        class_inv_remap[20] = 98
        class_strings[98] = "inserted-object"
        class_ignore_dict[20] = False
        nr_classes = 21
    else:
        class_remap[98] = 0
        nr_classes = 20
    
    ignore_list = [int(cl) for cl, ign in class_ignore_dict.items() if ign and int(cl) < nr_classes]

    # Make remap lookup table
    maxkey = max(class_remap.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in class_remap.items():
        remap_lut[key] = data

    # Collect all label and prediction files from the specified directories
    # Original GT (from SemanticKITTI format)
    orig_gt_path = os.path.join(FLAGS.original_gt, "labels")
    # Original Predictions (from RQ2 format: sequences/XX/predictions/)
    # Need to find the correct sequence directory
    orig_pred_seqs = glob.glob(os.path.join(FLAGS.original_pred, "sequences", "*/predictions"))
    
    # Transformed GT (from SemanticKITTI format)
    trans_gt_path = os.path.join(FLAGS.transformed_gt, "labels")
    # Transformed Predictions
    trans_pred_seqs = glob.glob(os.path.join(FLAGS.transformed_pred, "sequences", "*/predictions"))
    
    # Check GT paths exist
    if not os.path.exists(orig_gt_path):
        print(f"ERROR: Original GT path does not exist: {orig_gt_path}")
        quit()
    if not os.path.exists(trans_gt_path):
        print(f"ERROR: Transformed GT path does not exist: {trans_gt_path}")
        quit()
    
    # Collect GT files
    original_gt_files = sorted(glob.glob(os.path.join(orig_gt_path, "*.label")))
    transformed_gt_files = sorted(glob.glob(os.path.join(trans_gt_path, "*.label")))
    
    # Collect prediction files (from all sequences)
    original_pred_files = []
    for pred_seq_path in orig_pred_seqs:
        original_pred_files.extend(sorted(glob.glob(os.path.join(pred_seq_path, "*.label"))))
    
    transformed_pred_files = []
    for pred_seq_path in trans_pred_seqs:
        transformed_pred_files.extend(sorted(glob.glob(os.path.join(pred_seq_path, "*.label"))))
    
    # Sort prediction files
    original_pred_files = sorted(original_pred_files)
    transformed_pred_files = sorted(transformed_pred_files)
    
    print(f"Original GT files: {len(original_gt_files)}")
    print(f"Original Pred files: {len(original_pred_files)}")
    print(f"Transformed GT files: {len(transformed_gt_files)}")
    print(f"Transformed Pred files: {len(transformed_pred_files)}")
    
    # Validate file counts
    if len(original_gt_files) == 0:
        print(f"ERROR: No GT files found in {orig_gt_path}")
        quit()
    if len(transformed_gt_files) == 0:
        print(f"ERROR: No GT files found in {trans_gt_path}")
        quit()
    if len(original_pred_files) == 0:
        print(f"ERROR: No prediction files found in {FLAGS.original_pred}")
        quit()
    if len(transformed_pred_files) == 0:
        print(f"ERROR: No prediction files found in {FLAGS.transformed_pred}")
        quit()
    
    if len(original_gt_files) != len(original_pred_files):
        print(f"WARNING: Original GT files ({len(original_gt_files)}) != Pred files ({len(original_pred_files)})")
    if len(transformed_gt_files) != len(transformed_pred_files):
        print(f"WARNING: Transformed GT files ({len(transformed_gt_files)}) != Pred files ({len(transformed_pred_files)})")
    
    # Prepare output
    if os.path.exists(FLAGS.output):
        os.remove(FLAGS.output)
    
    save_to_log(FLAGS.output, "=" * 80)
    save_to_log(FLAGS.output, "DIVERSITY METRICS EVALUATION")
    save_to_log(FLAGS.output, f"Original GT: {FLAGS.original_gt}")
    save_to_log(FLAGS.output, f"Original Pred: {FLAGS.original_pred}")
    save_to_log(FLAGS.output, f"Transformed GT: {FLAGS.transformed_gt}")
    save_to_log(FLAGS.output, f"Transformed Pred: {FLAGS.transformed_pred}")
    save_to_log(FLAGS.output, "=" * 80)
    
    # ===== DATA DIVERSITY =====
    save_to_log(FLAGS.output, "\n" + "=" * 80)
    save_to_log(FLAGS.output, "DATA DIVERSITY (GT Label Distribution Comparison)")
    save_to_log(FLAGS.output, "=" * 80)
    
    # Get GT distributions
    orig_gt_dist, orig_gt_counts = get_label_distribution(original_gt_files, remap_lut, nr_classes)
    trans_gt_dist, trans_gt_counts = get_label_distribution(transformed_gt_files, remap_lut, nr_classes)
    
    # Compute data diversity metrics
    data_metrics = compute_diversity_metrics(orig_gt_dist, trans_gt_dist)
    
    save_to_log(FLAGS.output, f"Original GT total points: {orig_gt_counts.sum():,}")
    save_to_log(FLAGS.output, f"Transformed GT total points: {trans_gt_counts.sum():,}")
    save_to_log(FLAGS.output, "")
    save_to_log(FLAGS.output, "Data Diversity Metrics:")
    save_to_log(FLAGS.output, f"  Kullback-Leibler (KL):    {data_metrics['KL']:.6f}")
    save_to_log(FLAGS.output, f"  Jensen-Shannon (JS):      {data_metrics['JS']:.6f}")
    save_to_log(FLAGS.output, f"  Wasserstein Distance (WD): {data_metrics['WD']:.6f}")
    save_to_log(FLAGS.output, f"  Hellinger Distance (HD):   {data_metrics['HD']:.6f}")
    save_to_log(FLAGS.output, "")
    
    # ===== FAULT DIVERSITY =====
    save_to_log(FLAGS.output, "=" * 80)
    save_to_log(FLAGS.output, "FAULT DIVERSITY (FP Error Distribution Comparison)")
    save_to_log(FLAGS.output, "=" * 80)
    
    # Get FP distributions
    orig_fp_dist, orig_fp_counts = get_fp_distribution(
        original_gt_files, original_pred_files, remap_lut, nr_classes, ignore_list,
        apply_relaxation=FLAGS.include_inserted_object
    )
    trans_fp_dist, trans_fp_counts = get_fp_distribution(
        transformed_gt_files, transformed_pred_files, remap_lut, nr_classes, ignore_list,
        apply_relaxation=FLAGS.include_inserted_object
    )
    
    # Compute fault diversity metrics
    fault_metrics = compute_diversity_metrics(orig_fp_dist, trans_fp_dist)
    
    save_to_log(FLAGS.output, f"Original FP total points: {orig_fp_counts.sum():,}")
    save_to_log(FLAGS.output, f"Transformed FP total points: {trans_fp_counts.sum():,}")
    save_to_log(FLAGS.output, "")
    save_to_log(FLAGS.output, "Fault Diversity Metrics:")
    save_to_log(FLAGS.output, f"  Kullback-Leibler (KL):    {fault_metrics['KL']:.6f}")
    save_to_log(FLAGS.output, f"  Jensen-Shannon (JS):      {fault_metrics['JS']:.6f}")
    save_to_log(FLAGS.output, f"  Wasserstein Distance (WD): {fault_metrics['WD']:.6f}")
    save_to_log(FLAGS.output, f"  Hellinger Distance (HD):   {fault_metrics['HD']:.6f}")
    save_to_log(FLAGS.output, "")
    
    # ===== DETAILED COMPARISON =====
    save_to_log(FLAGS.output, "=" * 80)
    save_to_log(FLAGS.output, "DETAILED DISTRIBUTION COMPARISON")
    save_to_log(FLAGS.output, "=" * 80)
    
    # GT Distribution comparison
    save_to_log(FLAGS.output, "\nGT Label Distribution:")
    save_to_log(FLAGS.output, f"{'Class':<4} {'Name':<20} {'Original':<15} {'Original %':<12} {'Transformed':<15} {'Transformed %':<12}")
    save_to_log(FLAGS.output, "-" * 85)
    
    for class_id in range(nr_classes):
        if class_id not in ignore_list and (orig_gt_counts[class_id] > 0 or trans_gt_counts[class_id] > 0):
            if class_id == 20 and FLAGS.include_inserted_object:
                class_name = "inserted-object"
            elif class_id in class_inv_remap:
                original_label = class_inv_remap[class_id]
                class_name = class_strings.get(original_label, f"class-{class_id}")
            else:
                class_name = f"class-{class_id}"
            
            orig_pct = (orig_gt_counts[class_id] / orig_gt_counts.sum() * 100) if orig_gt_counts.sum() > 0 else 0
            trans_pct = (trans_gt_counts[class_id] / trans_gt_counts.sum() * 100) if trans_gt_counts.sum() > 0 else 0
            
            save_to_log(FLAGS.output,
                f"{class_id:<4} {class_name:<20} "
                f"{orig_gt_counts[class_id]:<15,d} "
                f"{orig_pct:>10.2f}% "
                f"{trans_gt_counts[class_id]:<15,d} "
                f"{trans_pct:>10.2f}%"
            )
    
    # FP Distribution comparison
    save_to_log(FLAGS.output, "\nFP Error Distribution:")
    save_to_log(FLAGS.output, f"{'Class':<4} {'Name':<20} {'Original FP':<15} {'Original %':<12} {'Transformed FP':<15} {'Transformed %':<12}")
    save_to_log(FLAGS.output, "-" * 85)
    
    for class_id in range(nr_classes):
        if class_id not in ignore_list and (orig_fp_counts[class_id] > 0 or trans_fp_counts[class_id] > 0):
            if class_id == 20 and FLAGS.include_inserted_object:
                class_name = "inserted-object"
            elif class_id in class_inv_remap:
                original_label = class_inv_remap[class_id]
                class_name = class_strings.get(original_label, f"class-{class_id}")
            else:
                class_name = f"class-{class_id}"
            
            orig_fp_pct = (orig_fp_counts[class_id] / orig_fp_counts.sum() * 100) if orig_fp_counts.sum() > 0 else 0
            trans_fp_pct = (trans_fp_counts[class_id] / trans_fp_counts.sum() * 100) if trans_fp_counts.sum() > 0 else 0
            
            save_to_log(FLAGS.output,
                f"{class_id:<4} {class_name:<20} "
                f"{orig_fp_counts[class_id]:<15,d} "
                f"{orig_fp_pct:>10.2f}% "
                f"{trans_fp_counts[class_id]:<15,d} "
                f"{trans_fp_pct:>10.2f}%"
            )
    
    save_to_log(FLAGS.output, "")
    save_to_log(FLAGS.output, "=" * 80)
    save_to_log(FLAGS.output, "Evaluation completed successfully!")
    save_to_log(FLAGS.output, "=" * 80)
    
    print(f"\nResults saved to: {FLAGS.output}")

