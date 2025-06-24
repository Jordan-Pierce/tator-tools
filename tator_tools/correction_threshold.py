"""

This script provides utilities for calibrating YOLOv8 models using a YAML-driven dataset.
It includes dataset loading, anchor generation, bounding box utilities, and a calibration function
that suggests an optimal confidence threshold for post-processing.

"""

import os
import cv2
import yaml 
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import argparse


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def make_anchors_pytorch(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchor points and stride tensors for each feature map.
    """
    anchor_points = []
    stride_tensor = []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=feats[i].device) + grid_cell_offset
        sy = torch.arange(h, device=feats[i].device) + grid_cell_offset
        grid_x, grid_y = torch.meshgrid(sx, sy, indexing='ij')
        anchors = torch.stack((grid_x, grid_y), -1).view(-1, 2)
        anchor_points.append(anchors)
        stride_tensor.append(torch.full((anchors.shape[0], 1), stride, device=feats[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Decode distances to bounding boxes.
    """
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox_iou_pytorch(box1, box2, xywh=True, eps=1e-7):
    """
    Compute IoU between two sets of bounding boxes.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter / union


class YamlYOLODataset(Dataset):
    """
    A PyTorch Dataset that loads data based on a YOLOv8-style YAML configuration file.
    It aggregates images from multiple directories specified for a given data split.
    """

    def __init__(self, yaml_file, split='val', target_size=(640, 640)):
        self.target_size = target_size
        # Load and parse the YAML file
        with open(yaml_file, 'r') as f:
            self.data = yaml.safe_load(f)
        self.names = self.data['names']
        self.nc = self.data['nc']
        # Determine the base path for relative paths
        base_path = self.data.get('path', os.path.dirname(yaml_file))
        # Check if the requested split exists
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in YAML file. Available splits: {list(self.data.keys())}")
        # Aggregate all image paths from the directories listed in the split
        self.img_paths = []
        img_dirs = self.data[split]
        if isinstance(img_dirs, str):
            img_dirs = [img_dirs]
        for img_dir in img_dirs:
            # Handle both absolute and relative paths
            if not os.path.isabs(img_dir):
                img_dir = os.path.join(base_path, img_dir)
            if not os.path.isdir(img_dir):
                print(f"WARNING: Directory not found, skipping: {img_dir}")
                continue
            for filename in os.listdir(img_dir):
                if filename.lower().endswith((
                        '.jpg', '.png', '.jpeg',
                )):
                    self.img_paths.append(os.path.join(img_dir, filename))
        if not self.img_paths:
            print(f"WARNING: No images found for split '{split}' based on YAML file '{yaml_file}'")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.target_size)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        # Intelligently find the corresponding label file
        # Assumes a structure like .../split/images/ -> .../split/labels/
        label_path = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
        label_path = os.path.splitext(label_path)[0] + '.txt'
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([float(p) for p in parts])
        labels = torch.tensor(boxes) if boxes else torch.zeros((0, 5))
        target = {"cls": labels[:, 0], "bboxes": labels[:, 1:]} if labels.numel() > 0 else \
                 {"cls": torch.tensor([]), "bboxes": torch.tensor([])}
        return img_tensor, target


def collate_fn(batch):
    """
    Collate function for DataLoader to batch images and targets.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    batched_targets = {'batch_idx': [], 'cls': [], 'bboxes': []}
    for i, target in enumerate(targets):
        if target['cls'].numel() > 0:
            batched_targets['batch_idx'].append(torch.full_like(target['cls'], i, dtype=torch.float))
            batched_targets['cls'].append(target['cls'])
            batched_targets['bboxes'].append(target['bboxes'])
    for key in batched_targets:
        if batched_targets[key]:
            batched_targets[key] = torch.cat(batched_targets[key], 0)
        else:
            batched_targets[key] = torch.tensor([])
    return images, batched_targets


def find_optimal_correction_thresh_pytorch(
    model,
    yaml_file,
    split='val',
    batch_size=16,
    iou_thresh_for_match=0.5,
    iou_thresh_for_bg=0.1
):
    """
    Finds the optimal threshold using a YAML-driven dataset.
    Args:
        model: PyTorch model (YOLOv8)
        yaml_file: Path to YAML config
        split: Data split to use (default 'val')
        batch_size: Batch size for DataLoader
        iou_thresh_for_match: IoU threshold for positive match
        iou_thresh_for_bg: IoU threshold for background
    Returns:
        suggested_thresh: Suggested confidence threshold (float)
    """
    device = next(model.parameters()).device
    model.eval()
    # 1. Create the YAML-driven PyTorch dataloader
    dataset = YamlYOLODataset(yaml_file=yaml_file, split=split)
    if len(dataset) == 0:
        return 0.3  # Return default if no data found
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
    # 2. Extract model properties
    nc = model.nc
    reg_max = model.model[-1].reg_max
    strides = model.model[-1].stride
    true_negative_scores = []
    false_negative_proxy_scores = []
    # 3. The main calibration loop
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Analyzing '{split}' split"):
            images = images.to(device)
            preds_raw = model(images)
            feats = preds_raw[1] if isinstance(preds_raw, tuple) else preds_raw
            anchor_points, stride_tensor = make_anchors_pytorch(feats, strides)
            pred_distri, pred_scores = torch.cat([
                xi.view(images.shape[0], nc + reg_max * 4, -1) for xi in feats
            ], 2).permute(0, 2, 1).contiguous().split((reg_max * 4, nc), 2)
            # Get original shape: [batch_size, num_anchors, channels]
            b, a, c = pred_distri.shape
            # 1. Reshape to isolate the reg_max dimension for each of the 4 coordinates.
            # Shape becomes: [batch_size, num_anchors, 4, reg_max]
            pred_distri_reshaped = pred_distri.view(b, a, 4, reg_max)
            # 2. Apply softmax over the reg_max dimension (dim=3) and matrix multiply
            #    with the projection vector [0, 1, ..., 15] to get the distance values.
            proj = torch.arange(reg_max, device=device, dtype=torch.float)
            distance_values = pred_distri_reshaped.softmax(3).matmul(proj)  # Result shape: [b, a, 4]
            # 3. Repeat anchor points for each image in the batch and decode the distances.
            #    The distance_values tensor must be reshaped to [b*a, 4] to match the anchors.
            tiled_anchor_points = anchor_points.repeat(b, 1)
            box_decoded = dist2bbox(distance_values.view(-1, 4), tiled_anchor_points, xywh=False)
            pd_bboxes_all_anchors = (box_decoded.view(images.shape[0], -1, 4) * stride_tensor).cpu()
            pd_scores_all_anchors = pred_scores.sigmoid().cpu()
            batch_idx_gt = targets['batch_idx'].long()
            bboxes_gt = targets['bboxes']
            for i in range(images.shape[0]):
                mask = batch_idx_gt == i
                gt_boxes_img = bboxes_gt[mask]
                pd_scores_img = pd_scores_all_anchors[i]
                pd_bboxes_img = pd_bboxes_all_anchors[i]
                if gt_boxes_img.numel() > 0:
                    gt_boxes_img_xyxy = torch.cat((gt_boxes_img[:, :2] - gt_boxes_img[:, 2:] / 2,
                                                   gt_boxes_img[:, :2] + gt_boxes_img[:, 2:] / 2), 1)
                    gt_boxes_img_xyxy *= images.shape[2]
                if gt_boxes_img.numel() == 0:
                    max_scores_per_anchor, _ = pd_scores_img.max(dim=1)
                    true_negative_scores.extend(max_scores_per_anchor[max_scores_per_anchor > 0.01].tolist())
                    continue
                iou_matrix = bbox_iou_pytorch(
                    pd_bboxes_img.unsqueeze(1), gt_boxes_img_xyxy.unsqueeze(0), xywh=False
                )
                max_iou_per_gt, _ = iou_matrix.max(dim=0)
                fn_mask = max_iou_per_gt < iou_thresh_for_match
                missed_gt_boxes = gt_boxes_img_xyxy[fn_mask]
                if missed_gt_boxes.numel() > 0:
                    missed_gt_centers = (missed_gt_boxes[:, :2] + missed_gt_boxes[:, 2:]) / 2.0
                    dist_matrix = torch.cdist(
                        missed_gt_centers, anchor_points.cpu() * stride_tensor.cpu()
                    )
                    closest_anchor_indices = dist_matrix.argmin(dim=1)
                    proxy_scores, _ = pd_scores_img[closest_anchor_indices].max(dim=1)
                    false_negative_proxy_scores.extend(proxy_scores.tolist())
                max_iou_per_pred, _ = iou_matrix.max(dim=1)
                tn_mask = max_iou_per_pred < iou_thresh_for_bg
                tn_pred_scores, _ = pd_scores_img[tn_mask].max(dim=1)
                true_negative_scores.extend(tn_pred_scores[tn_pred_scores > 0.01].tolist())
    # 4. Reporting
    print("✅ Analysis complete. Generating report...")
    if not true_negative_scores:
        print("WARNING: No true negative scores collected. Cannot suggest a threshold.")
        return 0.3
    plt.figure(figsize=(12, 7))
    sns.kdeplot(true_negative_scores, label='True Negative Scores (Background)', fill=True, clip=(0, 1), bw_adjust=0.2)
    if false_negative_proxy_scores:
        sns.kdeplot(
            false_negative_proxy_scores,
            label='False Negative Proxy Scores (Missed Objects)',
            fill=True,
            clip=(0, 1),
            bw_adjust=0.2
        )
    suggested_thresh = np.percentile(np.array(true_negative_scores), 99.5)
    plt.axvline(suggested_thresh, color='r', linestyle='--', label=f'Suggested Threshold ({suggested_thresh:.3f})')
    plt.title(f'Confidence Score Distributions - Split: {split}', fontsize=16, fontweight='bold')
    plt.xlabel('Model Confidence Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    report_path = f"pytorch_correction_thresh_analysis_{split}.png"
    plt.savefig(report_path)
    plt.close()
    print(f"📊 Report saved to {report_path}")
    print(f"🎯 RECOMMENDED THRESHOLD: {suggested_thresh:.4f}")
    return suggested_thresh


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """
    Main entry point for running the calibration script.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Correction Threshold Calibration")
    parser.add_argument('--data', type=str, required=True, help='Path to the YAML data config file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the YOLOv8 model weights')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to use (default: val)')
    args = parser.parse_args()

    try:
        model_path = args.model
        data_yaml = args.data
        split = args.split

        print(f"Loading YOLOv8 model from: {model_path}")
        yolo_obj = YOLO(model_path)
        pytorch_model = yolo_obj.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Starting calibration using YAML config: {data_yaml} (split: {split})")
        threshold = find_optimal_correction_thresh_pytorch(
            model=pytorch_model,
            yaml_file=data_yaml,
            split=split  # Specify which split to use for calibration
        )
        print(f"\n✅ SUCCESS! The suggested correction threshold is: {threshold:.4f}")

    except Exception:
        import traceback
        print("\n❌ ERROR: An error occurred during the process.")
        traceback.print_exc()


if __name__ == '__main__':
    main()