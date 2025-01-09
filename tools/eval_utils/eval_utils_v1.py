import pickle
import time
import json

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.compute_3d_iou import compute_3d_iou

def categorize_errors(gt_boxes, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
    errors = {
        'missed': 0,
        'false_positive': 0,
        'localization': 0,
        'classification': 0,
        'duplicate': 0
    }

    # Handle empty predictions
    if len(pred_boxes) == 0:
        errors['missed'] += len(gt_boxes)
        return errors

    # Match predictions with ground truth
    iou_matrix = compute_3d_iou(gt_boxes, pred_boxes)
    matched_gt = set()
    matched_pred = set()

    for gt_idx, gt_box in enumerate(gt_boxes):
        max_iou = 0
        best_pred = -1
        for pred_idx, pred_box in enumerate(pred_boxes):
            if pred_idx in matched_pred:
                continue
            iou = iou_matrix[gt_idx, pred_idx]
            if iou > max_iou:
                max_iou = iou
                best_pred = pred_idx

        if max_iou >= iou_threshold:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred)
            # Check for localization error
            if pred_labels[best_pred] != gt_box[-1]:  # Ensure this is the correct label index
                errors['classification'] += 1
            elif max_iou < 0.75:  # Consider making this configurable
                errors['localization'] += 1
        else:
            errors['missed'] += 1

    # Count false positives and duplicates
    for pred_idx in range(len(pred_boxes)):
        if pred_idx not in matched_pred:
            errors['false_positive'] += 1
        else:
            count = sum([1 for gt_idx in range(len(gt_boxes)) if iou_matrix[gt_idx, pred_idx] >= iou_threshold])
            if count > 1:
                errors['duplicate'] += count - 1

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

    error_metrics = {
        'missed': 0,
        'false_positive': 0,
        'localization': 0,
        'classification': 0,
        'duplicate': 0
    }

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
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
        gt_boxes = batch_dict['gt_boxes']
        for batch_idx in range(len(pred_dicts)):
            pred_dict = pred_dicts[batch_idx]
            pred_boxes = pred_dict['pred_boxes']
            pred_scores = pred_dict['pred_scores']
            pred_labels = pred_dict['pred_labels']

            # Convert to numpy and store
            all_gt_boxes.append(gt_boxes[batch_idx].cpu().numpy())
            all_pred_boxes.append(pred_boxes.cpu().numpy())
            all_pred_scores.append(pred_scores.cpu().numpy())
            all_pred_labels.append(pred_labels.cpu().numpy())

            # Calculate errors for this batch element
            errors = categorize_errors(
                gt_boxes=gt_boxes[batch_idx].cpu().numpy(),
                pred_boxes=pred_boxes.cpu().numpy(),
                pred_scores=pred_scores.cpu().numpy(),
                pred_labels=pred_labels.cpu().numpy()
            )
            for key in error_metrics.keys():
                error_metrics[key] += errors[key]

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # Concatenate all results
    all_gt_boxes = np.concatenate(all_gt_boxes, axis=0)
    all_pred_boxes = np.concatenate(all_pred_boxes, axis=0)
    all_pred_scores = np.concatenate(all_pred_scores, axis=0)
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)

    # Log error statistics
    logger.info('Error Analysis:')
    for key, value in error_metrics.items():
        logger.info(f'{key.capitalize()}: {value}')

    # Save error metrics
    with open(result_dir / 'error_metrics.json', 'w') as json_file:
        json.dump(error_metrics, json_file)

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

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
