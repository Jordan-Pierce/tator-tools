import numpy as np
import ultralytics.data.build as detection_build
from ultralytics.data.dataset import YOLODataset

class WeightedInstanceDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """
        super(WeightedInstanceDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

detection_build.YOLODataset = WeightedInstanceDataset


import torch
import torch.nn as nn
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import bbox_iou
from typing import Any, Dict, Tuple
from collections import deque
import numpy as np

# ---------------------------------------------------------------------------
# 1. Custom Assigner (with Negative Label Correction)
# ---------------------------------------------------------------------------

class CustomAssigner(nn.Module):
    """
    A custom task-aligned assigner that implements "Negative Label Correction".
    It drops background samples where the model predicts an object with high confidence,
    assuming these are unlabeled positives from a partially-labeled dataset.
    This prevents the model from being penalized for finding new, correct objects.
    """

    def __init__(self, topk: int = 10, num_classes: int = 80, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9, 
                 correction_thresh: float = 0.3):
        """
        Initialize the CustomAssigner.
        
        Args:
            topk (int): The number of top candidates to consider for assignment.
            num_classes (int): The number of object classes.
            alpha (float): The alpha parameter for the alignment metric.
            beta (float): The beta parameter for the alignment metric.
            eps (float): A small value to prevent division by zero.
            correction_thresh (float): Confidence threshold. Background samples with a max class score
                                     ABOVE this value will be dropped from the loss calculation.
                                     A value of 0.0 disables this feature.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
        # --- Our custom parameter for Negative Label Correction ---
        self.correction_thresh = correction_thresh
        
        if RANK in (-1, 0):
            print(f"✅ CustomAssigner initialized.")
            if self.correction_thresh > 0.0:
                print(f"   - Negative Label Correction: ENABLED (threshold = {self.correction_thresh})")
            else:
                print(f"   - Negative Label Correction: DISABLED")

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Computes the assignment and then applies our Negative Label Correction logic.
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            # Perform the standard assignment logic by calling the internal _forward method
            results = self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            LOGGER.warning("CUDA OutOfMemoryError in CustomAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            results = self._forward(*cpu_tensors)
            results = tuple(t.to(device) for t in results)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = results

        if self.training and self.correction_thresh > 0.0:
            # --- NEGATIVE LABEL CORRECTION LOGIC ---
            
            # 1. Identify all initial background regions
            bg_mask = ~fg_mask
            bg_indices = torch.where(bg_mask)

            if bg_indices[0].numel() > 0:
                # 2. Get the model's multi-class prediction scores for these background regions
                bg_scores_all_classes = pd_scores[bg_indices]
                
                # 3. Find the max score for each background anchor (its "objectness" confidence)
                bg_max_scores, _ = bg_scores_all_classes.max(dim=-1)

                # 4. Identify which samples have a confidence ABOVE our correction threshold.
                # These are the samples we suspect are unlabeled positives.
                mask_to_drop = bg_max_scores > self.correction_thresh
                
                # 5. Get the original indices of the samples to drop.
                batch_idx_to_drop = bg_indices[0][mask_to_drop]
                anchor_idx_to_drop = bg_indices[1][mask_to_drop]

                # 6. Exclude these suspected unlabeled positives from the loss calculation
                # by setting their target score to -1.
                target_scores[batch_idx_to_drop, anchor_idx_to_drop] = -1.0

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    # --- ALL HELPER METHODS ARE COPIED DIRECTLY FROM THE ORIGINAL ULTRALYTICS IMPLEMENTATION ---

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Internal forward pass for assignment."""
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select top-k candidates."""
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, bboxes and scores."""
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_labels.clamp_(0)
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """Select anchors in ground truth boxes."""
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """Handle multiple gt assignments."""
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


# ---------------------------------------------------------------------------
# 2. Convergence Monitor & Bounds Calculator
# ---------------------------------------------------------------------------

class ConvergenceMonitor:
    """
    Monitors training convergence and automatically determines threshold bounds
    based on confidence score distributions and training dynamics.
    """
    
    def __init__(self, window_size: int = 10, min_epochs_for_bounds: int = 5, tuning_profile: Dict = None):
        """
        Initialize the monitor.
        
        Args:
            window_size (int): Number of epochs to consider for loss trends.
            min_epochs_for_bounds (int): Minimum epochs of data needed before calculating bounds.
            tuning_profile (Dict): A dictionary containing parameters for tuning NLC aggressiveness.
        """
        self.window_size = window_size
        self.min_epochs_for_bounds = min_epochs_for_bounds
        
        # --- NEW: Store the tuning profile ---
        # Provide a default conservative profile in case none is passed
        if tuning_profile is None:
            if RANK in (-1, 0):
                LOGGER.warning("ConvergenceMonitor initialized without a tuning profile. Using default.")
            tuning_profile = {'tp_quantile': 0.10, 'max_thresh_min_cap': 0.20}
        self.tuning_profile = tuning_profile
        
        # Loss tracking for convergence detection
        self.loss_history = deque(maxlen=window_size)
        self.val_loss_history = deque(maxlen=window_size)
        
        # Confidence distribution tracking
        self.tp_conf_history = deque(maxlen=window_size)
        self.fp_conf_history = deque(maxlen=window_size)
        self.bg_conf_history = deque(maxlen=window_size)
        
        # Bounds calculation
        self.auto_min_thresh = 0.05  # Initial fallback
        self.auto_max_thresh = 0.50  # Initial fallback
        self.bounds_calculated = False
        
        # Convergence state
        self.convergence_factor = 1.0  # Multiplier for threshold adjustment
        
    def update_loss(self, train_loss: float, val_loss: float = None):
        """Update loss history for convergence monitoring."""
        self.loss_history.append(train_loss)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
    
    def update_confidence_stats(self, tp_confs: torch.Tensor, fp_confs: torch.Tensor, bg_confs: torch.Tensor):
        """Update confidence score distributions."""
        if tp_confs.numel() > 0:
            self.tp_conf_history.append(tp_confs.cpu())
        if fp_confs.numel() > 0:
            self.fp_conf_history.append(fp_confs.cpu())
        if bg_confs.numel() > 0:
            self.bg_conf_history.append(bg_confs.cpu())
    
    def calculate_automatic_bounds(self) -> Tuple[float, float]:
        """
        Automatically calculate threshold bounds based on confidence distributions.
        
        Returns:
            Tuple of (min_thresh, max_thresh)
        """
        if len(self.tp_conf_history) < self.min_epochs_for_bounds:
            return self.auto_min_thresh, self.auto_max_thresh
        
        all_tp_confs = torch.cat(list(self.tp_conf_history)) if self.tp_conf_history else torch.tensor([])
        all_bg_confs = torch.cat(list(self.bg_conf_history)) if self.bg_conf_history else torch.tensor([])
        
        # Calculate noise floor (minimum threshold)
        if all_bg_confs.numel() > 0:
            noise_floor = torch.quantile(all_bg_confs.float(), 0.95).item()
            self.auto_min_thresh = max(0.02, min(0.15, noise_floor + 0.02))
        
        # --- MODIFIED: Calculate ceiling (maximum threshold) using the tuning profile ---
        if all_tp_confs.numel() > 0:
            tp_quantile = self.tuning_profile.get('tp_quantile', 0.10)
            max_thresh_min_cap = self.tuning_profile.get('max_thresh_min_cap', 0.20)

            tp_floor = torch.quantile(all_tp_confs.float(), tp_quantile).item()
            # The -0.05 provides a small buffer below the weakest true positives
            self.auto_max_thresh = max(max_thresh_min_cap, min(0.85, tp_floor - 0.05))
        
        if self.auto_min_thresh >= self.auto_max_thresh:
            self.auto_max_thresh = self.auto_min_thresh + 0.15
        
        self.bounds_calculated = True
        return self.auto_min_thresh, self.auto_max_thresh
    
    def get_convergence_factor(self) -> float:
        """
        Calculate convergence factor based on training dynamics.
        
        Returns:
            Factor to multiply threshold adjustments by:
            - > 1.0 when converging well (be more aggressive)
            - < 1.0 when struggling (be more conservative)
        """
        if len(self.loss_history) < 3:
            return 1.0
        
        recent_losses = list(self.loss_history)[-3:]
        loss_trend = (recent_losses[-1] - recent_losses[0]) / max(recent_losses[0], 1e-6)
        loss_std = np.std(recent_losses) / max(np.mean(recent_losses), 1e-6)
        
        if loss_trend < -0.01:
            self.convergence_factor = 1.2 if loss_std < 0.05 else 1.0
        elif loss_trend > 0.01:
            self.convergence_factor = 0.7
        else:
            self.convergence_factor = 1.1 if loss_std < 0.02 else 0.9
        
        return self.convergence_factor
    
    def should_increase_threshold(self) -> bool:
        """Determine if threshold should be increased based on convergence."""
        return self.get_convergence_factor() > 1.0
    
    def should_decrease_threshold(self) -> bool:
        """Determine if threshold should be decreased based on convergence."""
        return self.get_convergence_factor() < 0.9


# ---------------------------------------------------------------------------
# 3. Custom Loss Class (with Convergence Monitoring)
# ---------------------------------------------------------------------------

class CustomV8DetectionLoss(v8DetectionLoss):
    """
    A custom version of v8DetectionLoss that uses our CustomAssigner
    and tracks statistics for automated threshold management.
    """
    def __init__(self, model):
        super().__init__(model)
        if not hasattr(self, 'bbox_decode'):
            self.bbox_decode = self.decode_bboxes
        
        # Get the custom argument from the training configuration
        correction_thresh = getattr(model.args, 'correction_thresh', 0.45)
        
        # Replace the default assigner with our custom one
        self.assigner = CustomAssigner(topk=10, 
                                       num_classes=self.nc, 
                                       alpha=0.5, 
                                       beta=6.0, 
                                       correction_thresh=correction_thresh)
        
        # Initialize confidence tracking for bounds calculation
        self.conf_stats = {
            'tp_confs': [],
            'fp_confs': [],
            'bg_confs': []
        }
        
        if RANK in (-1, 0):
            print("✅ CustomV8DetectionLoss initialized with convergence monitoring.")

    def __call__(self, preds, batch):
        """
        Calculates the loss and collects confidence statistics.
        This version includes a definitive fix to prevent detached graph errors
        in batches with no positive samples.
        """
        # Initialize individual loss components to simple zero tensors
        box_loss = torch.zeros(1, device=self.device)[0]
        cls_loss = torch.zeros(1, device=self.device)[0]
        dfl_loss = torch.zeros(1, device=self.device)[0]

        # --- Standard setup code from previous version ---
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # --- Assignment ---
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)
        
        self._collect_confidence_stats(pred_scores.detach().sigmoid(), target_scores, fg_mask)
            
        num_pos = fg_mask.sum()
        
        # --- Loss Calculation (only if positives exist) ---
        if num_pos > 0:
            # Classification loss
            cls_mask = (target_scores != -1.0)
            loss_cls_unnormalized = self.bce(pred_scores[cls_mask], target_scores[cls_mask].to(dtype))
            cls_loss = loss_cls_unnormalized.sum() / num_pos
    
            # Bbox and DFL loss
            target_scores_sum = max(target_scores[target_scores > 0].sum(), 1)
            target_bboxes /= stride_tensor
            try:
                box_loss, dfl_loss = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
            except TypeError:
                box_loss, dfl_loss = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum)

        # Apply loss gains (hyperparameters)
        box_loss = box_loss * self.hyp.box
        cls_loss = cls_loss * self.hyp.cls
        dfl_loss = dfl_loss * self.hyp.dfl
        
        # Combine the individual losses into a single total loss tensor
        total_loss = box_loss + cls_loss + dfl_loss

        # --- THE ROBUST FIX ---
        # To prevent a detached graph in batches with no GT objects (where num_pos=0),
        # ensure the final loss is connected to the computation graph.
        # We do this by adding a trivial operation involving one of the model's predictions.
        # This has no effect on the loss value but establishes the gradient path.
        total_loss = total_loss + 0.0 * pred_scores.sum()

        # Return the total loss for backpropagation and the detached itemized losses for logging
        return total_loss * batch_size, torch.stack((box_loss, cls_loss, dfl_loss)).detach()
    
    def _collect_confidence_stats(self, pred_scores, target_scores, fg_mask):
        """Collect confidence statistics for automatic bounds calculation."""
        with torch.no_grad():
            # Get max confidence per anchor
            max_confs, _ = pred_scores.max(dim=-1)
            
            # True positives: where we have actual foreground assignments
            tp_mask = fg_mask.bool()
            if tp_mask.any():
                tp_confs = max_confs[tp_mask]
                self.conf_stats['tp_confs'].append(tp_confs.cpu())
            
            # Background: where target_scores are all 0 (not -1 from correction)
            bg_mask = (target_scores.sum(dim=-1) == 0) & (target_scores[..., 0] != -1)
            if bg_mask.any():
                bg_confs = max_confs[bg_mask]
                self.conf_stats['bg_confs'].append(bg_confs.cpu())
    
    def get_recent_confidence_stats(self, max_samples: int = 1000):
        """Get recent confidence statistics for bounds calculation."""
        stats = {}
        for key, conf_list in self.conf_stats.items():
            if conf_list:
                # Concatenate recent samples
                recent_confs = torch.cat(conf_list[-10:])  # Last 10 batches
                if recent_confs.numel() > max_samples:
                    # Sample to avoid memory issues
                    indices = torch.randperm(recent_confs.numel())[:max_samples]
                    recent_confs = recent_confs[indices]
                stats[key] = recent_confs
            else:
                stats[key] = torch.tensor([])
        return stats
    
# ---------------------------------------------------------------------------
# 4. Automated Custom Trainer
# ---------------------------------------------------------------------------

class CustomTrainer(DetectionTrainer):
    """
    Fully automated trainer with convergence-based scheduling, automatic
    threshold bounds determination, and recall-centric performance logging.
    """
    # --- NEW: Tuning profiles for different training goals ---
    offset = 0.10
    TUNING_PROFILES = {
        'very_conservative': {
            'tp_quantile': 0.15,                    # Use 15th percentile for TP floor (stricter)
            'max_thresh_min_cap': 0.15 + offset,    # Minimum cap for the upper bound
            'adjustment_step': 0.01,                # Smaller, more cautious steps
        },
        'conservative': {
            'tp_quantile': 0.10,                    # Use 10th percentile for TP floor (stricter)
            'max_thresh_min_cap': 0.20 + offset,    # Minimum cap for the upper bound
            'adjustment_step': 0.015,               # Smaller, more cautious steps
        },
        'balanced': {
            'tp_quantile': 0.05,                    # Use 5th percentile (recommended start)
            'max_thresh_min_cap': 0.25 + offset,    # Moderate cap
            'adjustment_step': 0.03,                # Moderate steps
        },
        'aggressive': {
            'tp_quantile': 0.01,                    # Use 1st percentile (very permissive)
            'max_thresh_min_cap': 0.30 + offset,    # High cap, allows for high thresholds
            'adjustment_step': 0.05,                # Large, reactive steps
        },
        'very_aggressive': {
            'tp_quantile': 0.005,                   # Use 0.5th percentile (extremely permissive)
            'max_thresh_min_cap': 0.35 + offset,    # Very high cap
            'adjustment_step': 0.1,                 # Very large steps
        },
        'static': {
            'tp_quantile': None,                    # Sentinel value indicating no adaptation
            'max_thresh_min_cap': None,
            'adjustment_step': 0.0,                 # No adjustments will be made
        },
    }

    def _setup_train(self, world_size):
        """Setup training with automated threshold management."""
        if RANK in (-1, 0):
            LOGGER.info("🚀 Starting CustomTrainer setup...")
            
        super()._setup_train(world_size)

        # --- NEW: Select and log the tuning profile ---
        self.tuning_mode = "balanced"  # Default tuning mode

        self.tuning_profile = self.TUNING_PROFILES[self.tuning_mode]
        
        if RANK in (-1, 0):
            LOGGER.info(f"   - Tuning Profile: '{self.tuning_mode.upper()}'")
            if self.tuning_mode != 'static':
                LOGGER.info(f"     - TP Quantile for Bounds: {self.tuning_profile['tp_quantile']}")
                LOGGER.info(f"     - Scheduler Step Size: {self.tuning_profile['adjustment_step']}")

        # Replace with custom loss
        self.model.criterion = CustomV8DetectionLoss(self.model)
        self.model.criterion.assigner.to(self.device)
        
        # Initialize convergence monitor, passing the selected profile
        self.convergence_monitor = ConvergenceMonitor(
            window_size=getattr(self.args, 'convergence_window', 10),
            min_epochs_for_bounds=getattr(self.args, 'min_epochs_for_bounds', 3),
            tuning_profile=self.tuning_profile
        )
        
        # Variables for advanced performance logging
        self.best_map50 = 0.0
        self.last_map50 = 0.0
        
        # Register callbacks
        self.add_callback("on_train_epoch_start", self._automated_threshold_update)
        self.add_callback("on_fit_epoch_end", self._update_convergence_stats)
        self.add_callback("on_fit_epoch_end", self._log_epoch_performance_story)
        
        if RANK in (-1, 0):
            LOGGER.info("✅ CustomTrainer setup complete!")

    def _automated_threshold_update(self, *args, **kwargs):
        """Automated threshold update with convergence-based scheduling."""
        if self.tuning_mode == 'static':
            if self.epoch == 0 and RANK in (-1, 0):
                # Log this once at the beginning of the first epoch for clarity
                initial_thresh = self.model.criterion.assigner.correction_thresh
                LOGGER.info(f"🔄 Static Profile enabled. Correction threshold will remain fixed at {initial_thresh:.4f} for the entire run.")
            return
        
        current_epoch = self.epoch + 1
        if not isinstance(self.model.criterion.assigner, CustomAssigner):
            return
        
        current_thresh = self.model.criterion.assigner.correction_thresh
        if RANK in (-1, 0):
            LOGGER.info(f"\n🔄 Epoch {current_epoch}: Automated threshold update (Current: {current_thresh:.4f})")
        
        min_thresh, max_thresh = self.convergence_monitor.auto_min_thresh, self.convergence_monitor.auto_max_thresh
        if not self.convergence_monitor.bounds_calculated:
            if RANK in (-1, 0):
                LOGGER.info(f"   - Awaiting sufficient data for bounds. Threshold unchanged.")
            return

        if current_epoch <= getattr(self.args, 'adaptive_start_epoch', 3):
            new_thresh = min_thresh
            if RANK in (-1, 0):
                LOGGER.info(f"   - Warm-up phase. Using min threshold: {new_thresh:.4f}")
        else:
            convergence_factor = self.convergence_monitor.get_convergence_factor()
            
            # --- MODIFIED: Use adjustment step from the profile ---
            adjustment_step_size = self.tuning_profile['adjustment_step']
            adjustment = adjustment_step_size * (convergence_factor - 1.0)
            new_thresh = current_thresh + adjustment
            
            if RANK in (-1, 0):
                LOGGER.info(f"   - Adaptive Phase: Conv. Factor={convergence_factor:.2f}, Adjustment={adjustment:+.4f}, Bounds=[{min_thresh:.4f}, {max_thresh:.4f}]")
            
            new_thresh = np.clip(new_thresh, min_thresh, max_thresh).item()

        if abs(new_thresh - current_thresh) > 1e-5:
            self.model.criterion.assigner.correction_thresh = new_thresh
            if RANK in (-1, 0):
                LOGGER.info(f"   - ✅ New threshold applied: {new_thresh:.4f}")
        else:
            if RANK in (-1, 0):
                LOGGER.info(f"   - Threshold remains unchanged.")

    def _log_epoch_performance_story(self, *args, **kwargs):
        """
        Analyzes epoch metrics to log a descriptive "story" of the training progress.
        It first logs the P/R strategy, then logs mAP milestones and warnings.
        """
        if RANK in (-1, 0) and self.metrics:
            # Define keys for the metrics we need
            precision_key = 'metrics/precision(B)'
            recall_key = 'metrics/recall(B)'
            map50_key = 'metrics/mAP50(B)'

            # Ensure all required metrics are available
            if not all(k in self.metrics for k in [precision_key, recall_key, map50_key]):
                return
            
            p = self.metrics[precision_key]
            r = self.metrics[recall_key]
            map50 = self.metrics[map50_key]

            # --- Block 1: Log the model's "strategy" based on Precision vs. Recall ---
            # This has priority and describes the nature of the model's performance.
            # A small threshold (e.g., > 0.1) prevents logging during early, noisy epochs.
            if r > p * 1.1 and r > 0.1:
                LOGGER.info(f"🌟 High-Recall Epoch!      (P: {p:.3f}, R: {r:.3f}). Model is in 'discovery' mode.")
            elif p > r * 1.1 and p > 0.1:
                LOGGER.info(f"🎯 High-Precision Epoch!   (P: {p:.3f}, R: {r:.3f}). Model is in 'confident' mode.")
            elif abs(p - r) < p * 0.1 and p > 0.1: # Check for balance if not strongly skewed
                LOGGER.info(f"⚖️  Balanced Performance.   (P: {p:.3f}, R: {r:.3f}).")

            # --- Block 2: Log major milestones and health warnings based on mAP ---
            # This is logged in addition to the strategy message above, providing more context.
            if map50 > self.best_map50:
                LOGGER.info(f"🚀 New Best mAP50! {map50:.4f} (previously {self.best_map50:.4f})")
                self.best_map50 = map50
            elif self.epoch > 10 and map50 < self.best_map50 * 0.98 and self.best_map50 > 0.1:
                LOGGER.info(f"⚠️  Potential Overfitting! mAP50 dropped to {map50:.4f} from a peak of {self.best_map50:.4f}")
            elif self.epoch > 15 and abs(map50 - self.last_map50) < 0.001 and map50 > 0.1:
                LOGGER.info(f"⏳ Performance Stagnated. mAP50: {map50:.4f}")

            # --- Update the 'last epoch' metric for the next iteration's comparison ---
            self.last_map50 = map50

    def _update_convergence_stats(self, *args, **kwargs):
        """Update convergence monitoring statistics at the end of each epoch."""
        # Update loss history
        if hasattr(self, 'loss') and self.loss is not None:
            train_loss = float(self.loss.mean()) if hasattr(self.loss, 'mean') else float(self.loss)
            val_loss = self.metrics.get('val/box_loss', self.metrics.get('metrics/box_loss', None))
            if val_loss is not None:
                val_loss = float(val_loss.mean()) if hasattr(val_loss, 'mean') else float(val_loss)
            self.convergence_monitor.update_loss(train_loss, val_loss)
        
        # Update confidence statistics
        if hasattr(self.model.criterion, 'get_recent_confidence_stats'):
            stats = self.model.criterion.get_recent_confidence_stats()
            self.convergence_monitor.update_confidence_stats(
                stats.get('tp_confs', torch.tensor([])),
                stats.get('fp_confs', torch.tensor([])),
                stats.get('bg_confs', torch.tensor([]))
            )
            self.model.criterion.conf_stats = {'tp_confs': [], 'fp_confs': [], 'bg_confs': []}
        
        # Calculate automatic bounds periodically
        if (self.epoch + 1) % getattr(self.args, 'bounds_update_frequency', 5) == 0 and (self.epoch + 1) >= self.convergence_monitor.min_epochs_for_bounds:
            old_min, old_max = self.convergence_monitor.auto_min_thresh, self.convergence_monitor.auto_max_thresh
            new_min, new_max = self.convergence_monitor.calculate_automatic_bounds()
            
            if RANK in (-1, 0) and (abs(old_min - new_min) > 0.01 or abs(old_max - new_max) > 0.01):
                LOGGER.info(f"📊 Updated automatic bounds: [{old_min:.3f}, {old_max:.3f}] → [{new_min:.3f}, {new_max:.3f}]")
                
data_dir = "E:/JordanP/Click-a-Coral/data/reduced/MDBC_Transects_Coral_Sponges"
dataset = f"{data_dir}/YOLO_Detection_Dataset/data.yaml"
project = f"{data_dir}/results"

model = "E:/JordanP/Click-a-Coral/data/reduced/Season_3/results/yolov8n/weights/best.pt"
encoder = f"{data_dir}/results/yolov8n_classify/weights/best.pt"


args = dict(
    model=model,
    data=dataset,
    project=project,
    name="NL_MutiClass_Corrective_Pretrained_Weighted_w_Season_3_w_balanced",
    task='detect',
    epochs=100,
    patience=10,
    half=True,
    imgsz=640,
    single_cls=False,
    plots=True,
    batch=32,
    workers=0,
    save_period=1,
)

trainer = CustomTrainer(overrides=args)
trainer.train()