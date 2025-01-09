import numpy as np
from shapely.geometry import Polygon

def compute_3d_iou(gt_boxes, pred_boxes):
    """
    Computes the 3D IoU matrix between ground truth and predicted bounding boxes.

    Args:
        gt_boxes (np.ndarray): Ground truth boxes, shape (N_gt, 7).
                               Each box format: [x, y, z, l, w, h, theta].
        pred_boxes (np.ndarray): Predicted boxes, shape (N_pred, 7).
                                 Each box format: [x, y, z, l, w, h, theta].

    Returns:
        np.ndarray: IoU matrix of shape (N_gt, N_pred).
    """
    def rotate_bbox(bbox):
        """
        Computes the four corners of a 2D rotated bounding box in BEV.

        Args:
            bbox (list): Bounding box with [x, y, l, w, theta].

        Returns:
            np.ndarray: Array of shape (4, 2) with coordinates of the corners.
        """
        cx, cy, length, width, angle = bbox
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        half_l = length / 2
        half_w = width / 2

        # Define corners relative to the center
        corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])

        # Rotate and translate corners
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_corners = np.dot(corners, rotation_matrix) + np.array([cx, cy])

        return rotated_corners

    def compute_bev_iou_and_intersection(box1, box2):
        """
        Computes the 2D IoU in Bird's Eye View (BEV) for two rotated boxes.

        Args:
            box1 (np.ndarray): First box [x, y, l, w, theta].
            box2 (np.ndarray): Second box [x, y, l, w, theta].

        Returns:
            float: 2D IoU between the two boxes in BEV.
        """
        try:
            corners1 = rotate_bbox(box1)
            corners2 = rotate_bbox(box2)

            poly1 = Polygon(corners1)
            poly2 = Polygon(corners2)
            
            if not (poly1.is_valid and poly2.is_valid):
                return 0.0, 0.0
            
            intersection = poly1.intersection(poly2)
            inter_area = intersection.area if intersection.is_valid else 0.0
            union_area = poly1.area + poly2.area - inter_area

            iou = inter_area / max(union_area, 1e-5)
            return iou, inter_area
        except Exception:
            return 0.0, 0.0

    def compute_height_overlap(box1, box2):
        """
        Computes the height overlap between two 3D boxes.

        Args:
            box1 (np.ndarray): First box [z, h] where z is the center height and h is the height.
            box2 (np.ndarray): Second box [z, h] where z is the center height and h is the height.

        Returns:
            float: Height overlap.
        """
        z_min1 = box1[0] - box1[1] / 2
        z_max1 = box1[0] + box1[1] / 2
        z_min2 = box2[0] - box2[1] / 2
        z_max2 = box2[0] + box2[1] / 2

        overlap = max(0, min(z_max1, z_max2) - max(z_min1, z_min2))
        return overlap

    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]
    iou_matrix = np.zeros((num_gt, num_pred))

    for i in range(num_gt):
        for j in range(num_pred):
            # BEV IoU and intersection area
            bev_iou, bev_intersection = compute_bev_iou_and_intersection(
                [gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][3], gt_boxes[i][4], gt_boxes[i][6]],
                [pred_boxes[j][0], pred_boxes[j][1], pred_boxes[j][3], pred_boxes[j][4], pred_boxes[j][6]]
            )

            # Height overlap
            h_overlap = compute_height_overlap(
                [gt_boxes[i][2], gt_boxes[i][5]], 
                [pred_boxes[j][2], pred_boxes[j][5]]
            )

            # 3D IoU calculation
            intersection_vol = bev_intersection * h_overlap
            vol_gt = gt_boxes[i][3] * gt_boxes[i][4] * gt_boxes[i][5]
            vol_pred = pred_boxes[j][3] * pred_boxes[j][4] * pred_boxes[j][5]
            union_vol = vol_gt + vol_pred - intersection_vol

            iou_matrix[i, j] = intersection_vol / max(union_vol, 1e-5)

    return iou_matrix
