import pickle
import time
import json

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.compute_3d_iou import compute_3d_iou

def categorize_errors(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5, loc_threshold=0.1):
    """
    Categorizes detection errors according to the TIDE framework.

    Args:
        gt_boxes (np.ndarray): Ground truth boxes, shape (N_gt, 7).
        gt_labels (np.ndarray): Ground truth labels, shape (N_gt,).
        pred_boxes (np.ndarray): Predicted boxes, shape (N_pred, 7).
        pred_scores (np.ndarray): Predicted scores, shape (N_pred,).
        pred_labels (np.ndarray): Predicted labels, shape (N_pred,).
        iou_threshold (float): IoU threshold for matching.
        loc_threshold (float): Additional IoU threshold for localization error.

    Returns:
        dict: Dictionary containing counts of each error type.
    """
    errors = {
        'missed_gt': 0,
        'false_positive': 0,
        'duplicate_error': 0,
        'localization_error': 0,
        'classification_error': 0,
        'both_error': 0,
        'true_positive': 0
    }

    # Handle empty predictions
    if len(pred_boxes) == 0:
        errors['missed_gt'] += len(gt_boxes)
        return errors

    # Handle empty ground truths
    if len(gt_boxes) == 0:
        errors['false_positive'] += len(pred_boxes)
        return errors

    # Compute IoU matrix
    iou_matrix = compute_3d_iou(gt_boxes, pred_boxes)  # Shape: (N_gt, N_pred)

    # Initialize matches
    gt_matched = -np.ones(len(gt_boxes), dtype=int)    # -1 indicates unmatched
    pred_matched = -np.ones(len(pred_boxes), dtype=int)

    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes_sorted = pred_boxes[sorted_indices]
    pred_scores_sorted = pred_scores[sorted_indices]
    pred_labels_sorted = pred_labels[sorted_indices]
    iou_matrix_sorted = iou_matrix[:, sorted_indices]

    # Greedy matching
    for gt_idx in range(len(gt_boxes)):
        max_iou = 0
        max_pred_idx = -1
        for pred_idx in range(len(pred_boxes_sorted)):
            if pred_matched[pred_idx] != -1:
                continue
            iou = iou_matrix_sorted[gt_idx, pred_idx]
            if iou >= iou_threshold and iou > max_iou:
                max_iou = iou
                max_pred_idx = pred_idx
        if max_pred_idx != -1:
            gt_matched[gt_idx] = max_pred_idx
            pred_matched[max_pred_idx] = gt_idx

    # Error classification
    for gt_idx in range(len(gt_boxes)):
        if gt_matched[gt_idx] == -1:
            # Ground truth not matched
            errors['missed_gt'] += 1
        else:
            pred_idx = gt_matched[gt_idx]
            pred_label = pred_labels_sorted[pred_idx]
            gt_label = gt_labels[gt_idx]
            iou = iou_matrix_sorted[gt_idx, pred_idx]
            correct_class = pred_label == gt_label
            good_localization = iou >= (iou_threshold + loc_threshold)

            if correct_class and good_localization:
                # True positive
                errors['true_positive'] += 1
            elif not correct_class and not good_localization:
                errors['both_error'] += 1
            elif not correct_class:
                errors['classification_error'] += 1
            elif not good_localization:
                errors['localization_error'] += 1

    # Unmatched predictions (False Positives)
    for pred_idx in range(len(pred_boxes_sorted)):
        if pred_matched[pred_idx] == -1:
            errors['false_positive'] += 1

    # Duplicate detections: multiple predictions matched to the same ground truth
    gt_match_counts = np.bincount(gt_matched[gt_matched >= 0], minlength=len(gt_boxes))
    errors['duplicate_error'] += int(np.sum(gt_match_counts[gt_match_counts > 1] - 1))

    return errors

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    all_gt_boxes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    # Initialize error metrics
    errors = {
        'missed_gt': 0,
        'false_positive': 0,
        'background': 0,
        'localization_error': 0,
        'classification_error': 0,
        'both_error': 0,
        'duplicate_error': 0
    }
    
    logger.info(f'*************** EPOCH {epoch_id} EVALUATION *****************')
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos

        # Process each batch element's predictions
        gt_boxes = batch_dict['gt_boxes']  # shape: (batch_size, num_gt_boxes, 8)
        for batch_idx in range(len(pred_dicts)):
            pred_dict = pred_dicts[batch_idx]
            pred_boxes = pred_dict['pred_boxes'].cpu().numpy()  # (N_pred, 7)
            pred_scores = pred_dict['pred_scores'].cpu().numpy()  # (N_pred,)
            pred_labels = pred_dict['pred_labels'].cpu().numpy()  # (N_pred,)

            # Extract GT boxes and labels for this batch element
            gt_boxes_batch = gt_boxes[batch_idx].cpu().numpy()  # (N_gt, 8)
            gt_boxes_np = gt_boxes_batch[:, :7]  # (N_gt, 7)
            gt_labels = gt_boxes_batch[:, 7].astype(np.int32)  # (N_gt,)

            # Store for later use if needed
            all_gt_boxes.append(gt_boxes_np)
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_pred_labels.append(pred_labels)

            # Calculate errors for this batch element
            errors_batch = categorize_errors(
                gt_boxes=gt_boxes_np,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels
            )

            # Accumulate errors
            for key in errors_batch.keys():
                errors[key] += errors_batch[key]

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # Concatenate all results if needed
    if len(all_gt_boxes) > 0:
        all_gt_boxes = np.concatenate(all_gt_boxes, axis=0)
    if len(all_pred_boxes) > 0:
        all_pred_boxes = np.concatenate(all_pred_boxes, axis=0)
    if len(all_pred_scores) > 0:
        all_pred_scores = np.concatenate(all_pred_scores, axis=0)
    if len(all_pred_labels) > 0:
        all_pred_labels = np.concatenate(all_pred_labels, axis=0)

    # Log error statistics
    logger.info('Error Analysis:')
    for key, value in errors.items():
        logger.info(f'{key.capitalize()}: {value}')

    # Save error metrics
    with open(result_dir / 'error_metrics.json', 'w') as json_file:
        json.dump(errors, json_file)

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric_list = [metric]
        metric = common_utils.merge_results_dist(metric_list, world_size, tmpdir=result_dir / 'tmpdir')

        # Merge error metrics across processes
        errors_list = [errors]
        errors_merged = common_utils.merge_results_dist(errors_list, world_size, tmpdir=result_dir / 'tmpdir')
        # Sum errors
        errors = errors_merged[0]
        for e in errors_merged[1:]:
            for key in errors.keys():
                errors[key] += e[key]

    logger.info(f'*************** Performance of EPOCH {epoch_id} *****************')
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info(f'Generate label finished(sec_per_example: {sec_per_example:.4f} second).')

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key in metric[0].keys():
            for k in range(1, len(metric)):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric[f'recall_roi_{cur_thresh}'] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric[f'recall_rcnn_{cur_thresh}'] / max(gt_num_cnt, 1)
        logger.info(f'recall_roi_{cur_thresh}: {cur_roi_recall:.4f}')
        logger.info(f'recall_rcnn_{cur_thresh}: {cur_rcnn_recall:.4f}')
        ret_dict[f'recall/roi_{cur_thresh}'] = cur_roi_recall
        ret_dict[f'recall/rcnn_{cur_thresh}'] = cur_rcnn_recall

    total_pred_objects = sum(len(anno['name']) for anno in det_annos)
    logger.info(f'Average predicted number of objects({len(det_annos)} samples): {total_pred_objects / max(1, len(det_annos)):.3f}')

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
