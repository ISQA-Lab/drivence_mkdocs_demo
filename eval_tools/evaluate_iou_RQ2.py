#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# RQ2 Special Evaluation:
# - Excludes static background classes: road, sidewalk, parking, other-ground, building, terrain
# - Label 98 (inserted object) uses relaxed_dynamic_foreground standard:
#   predicted as any vehicle or human is considered correct

import argparse
import os
import yaml
import sys
import numpy as np
import torch

from modules.ioueval import iouEval
from common.laserscan import SemLaserScan

# possible splits
splits = ['train','valid','test']
def save_to_log(logdir,logfile,message):
    f = open(logdir+'/'+logfile, "a")
    f.write(message+'\n')
    f.close()
    return

def eval(test_sequences,splits,pred):
    # get scan paths
    scan_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        scan_paths = os.path.join(FLAGS.dataset, "sequences",
                                  str(sequence), "velodyne")
        # populate the scan names
        seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
        seq_scan_names.sort()
        scan_names.extend(seq_scan_names)

    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences",
                                   str(sequence), "labels")
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(FLAGS.predictions, "sequences",
                                  sequence, "predictions")
        # populate the label names
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)

    # check that I have the same number of files
    assert (len(label_names) == len(scan_names) and
            len(label_names) == len(pred_names))

    print("=" * 80)
    print("RQ2 EVALUATION MODE")
    print("Excluded classes: road, sidewalk, parking, other-ground, building, terrain")
    if FLAGS.include_inserted_object:
        print("Label 98: INCLUDED - relaxed_dynamic_foreground (vehicle or human is correct)")
    else:
        print("Label 98: EXCLUDED - mapped to unlabeled (ignored)")
    print("=" * 80)
    
    # open each file, get the tensor, and make the iou comparison
    for scan_file, label_file, pred_file in zip(scan_names, label_names, pred_names):
        print("evaluating label ", label_file, "with", pred_file)
        # open label
        label = SemLaserScan(project=False)
        label.open_scan(scan_file)
        label.open_label(label_file)
        u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_label_sem = u_label_sem[:FLAGS.limit]

        # open prediction
        pred = SemLaserScan(project=False)
        pred.open_scan(scan_file)
        pred.open_label(pred_file)
        u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_pred_sem = u_pred_sem[:FLAGS.limit]

        # SPECIAL HANDLING for label 98: relaxed_dynamic_foreground (only if included)
        if FLAGS.include_inserted_object:
            # Accept all vehicles: car (1), bicycle (2), motorcycle (3), truck (4), other-vehicle (5)
            # Accept all humans: person (6), bicyclist (7), motorcyclist (8)
            inserted_object_class = 20  # Label 98 mapped to class 20
            inserted_gt_mask = (u_label_sem == inserted_object_class)
            relaxed_pred = u_pred_sem.copy()
            
            dynamic_foreground_classes = [1, 2, 3, 4, 5, 6, 7, 8]
            for dyn_class in dynamic_foreground_classes:
                relaxed_pred[(inserted_gt_mask) & (u_pred_sem == dyn_class)] = inserted_object_class
        else:
            # Don't apply relaxation if not including inserted object
            relaxed_pred = u_pred_sem

        # add single scan to evaluation
        evaluator.addBatch(relaxed_pred, u_label_sem)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    print('{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=splits,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))

    save_to_log(FLAGS.predictions,'pred_RQ2.txt','{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=splits,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            # Handle the custom inserted object class
            if i == 20 and FLAGS.include_inserted_object:
                class_name = "inserted-object"
            else:
                class_name = class_strings[class_inv_remap[i]] if i in class_inv_remap else f"class-{i}"
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_name, jacc=jacc))
            save_to_log(FLAGS.predictions, 'pred_RQ2.txt', 
                       'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_name, jacc=jacc))

    # print for spreadsheet
    print("*" * 80)
    print("below can be copied straight for paper table")
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_iou_RQ2.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default="/home/atri/WD_Passport/semanticKITTI/dataset/",
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default="./RQ2_ng/1/insert_0",
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default. If no option is set'
             ' we look for the labels in the same directory as dataset'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        choices=["train", "valid", "test"],
        default="valid",
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
             ' evaluating single scan from aggregated pointcloud.'
             ' Defaults to %(default)s',
    )
    parser.add_argument(
        '--include_inserted_object',
        action='store_true',
        default=True,
        help='Include label 98 (inserted object) in evaluation. Default: True (include)',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset

    # print summary of what we will do
    print("*" * 80)
    print("RQ2 EVALUATION INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Limit: ", FLAGS.limit)
    print("Include inserted object (label 98): ", FLAGS.include_inserted_object)
    print("*" * 80)

    # assert split
    assert (FLAGS.split in splits)

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    
    # ADD CUSTOM MAPPING FOR LABEL 98 (inserted object) based on flag
    if FLAGS.include_inserted_object:
        class_remap[98] = 20  # Map label 98 to a new class 20
        class_inv_remap[20] = 98  # Inverse mapping
        class_ignore[20] = False  # Evaluate this class
        class_strings[98] = "inserted-object"  # Add name for display
        print("Including label 98 (mapped to class 20), will be evaluated")
    else:
        class_remap[98] = 0  # Map label 98 to unlabeled (ignore it)
        print("Excluding label 98 (mapped to unlabeled), will be ignored")
    
    # RQ2 SPECIFIC: Ignore static background classes
    # road (9), parking (10), sidewalk (11), other-ground (12), building (13), terrain (17)
    rq2_ignore_classes = [9, 10, 11, 12, 13, 17]
    for ignore_cls in rq2_ignore_classes:
        class_ignore[ignore_cls] = True
    
    nr_classes = len(class_inv_remap)
    print(f"Total classes: {nr_classes}")
    print(f"RQ2: Ignoring static background classes: {rq2_ignore_classes}")

    # make lookup table for mapping
    maxkey = 0
    for key, data in class_remap.items():
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in class_remap.items():
        try:
            remap_lut[key] = data
        except IndexError:
            print("Wrong key ", key)

    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            ignore_name = class_strings[class_inv_remap[x_cl]] if x_cl in class_inv_remap else f"class-{x_cl}"
            print("Ignoring xentropy class ", x_cl, f" [{ignore_name}] in IoU evaluation")

    # create evaluator
    device = torch.device("cpu")
    evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()

    # get test set
    if FLAGS.split is None:
        for splits in ('train','valid'):
            eval((DATA["split"][splits]),splits,FLAGS.predictions)
    else:
        eval(DATA["split"][FLAGS.split],splits,FLAGS.predictions)





