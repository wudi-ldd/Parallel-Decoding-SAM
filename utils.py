import logging

import cv2
import numpy as np
import torch

def dice_loss(preds, targets, smooth=1e-6):
    """Compute the soft Dice loss for binary segmentation logits.

    Args:
        preds (torch.Tensor): Raw logits of shape ``(B, 1, H, W)``.
        targets (torch.Tensor): Ground-truth masks broadcastable to logits.
        smooth (float): Numerical stability constant.

    Returns:
        torch.Tensor: Mean Dice loss over the batch.
    """
    preds = torch.sigmoid(preds)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss.mean()


def compute_loss(
    seg_head_logits,
    final_predictions,
    masks,
    loss_fn,
    seg_weight=1.0,
    sam_weight=1.0,
    loss_weights=(1, 1),
):
    """Compute combined segmentation-head and SAM losses."""
    seg_bce = loss_fn(seg_head_logits, masks)
    seg_dice = dice_loss(seg_head_logits, masks)
    seg_main_loss = loss_weights[0] * seg_bce + loss_weights[1] * seg_dice

    sam_bce = loss_fn(final_predictions, masks)
    sam_dice = dice_loss(final_predictions, masks)
    sam_main_loss = loss_weights[0] * sam_bce + loss_weights[1] * sam_dice

    total_loss = seg_weight * seg_main_loss + sam_weight * sam_main_loss

    return total_loss, seg_main_loss.item(), sam_main_loss.item()


def initialize_metrics():
    """Initialise counters for IoU/F1 metrics."""
    return {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'intersection': 0,
        'union': 0,
    }


def accumulate_metrics(preds, targets, global_metrics, threshold=0.5):
    """Update global metrics based on thresholded predictions."""
    preds_binary = (preds > threshold).astype(np.uint8)
    targets_binary = (targets > threshold).astype(np.uint8)

    tp = np.logical_and(preds_binary == 1, targets_binary == 1).sum()
    fp = np.logical_and(preds_binary == 1, targets_binary == 0).sum()
    fn = np.logical_and(preds_binary == 0, targets_binary == 1).sum()

    intersection = tp
    union = np.logical_or(preds_binary, targets_binary).sum()

    global_metrics['tp'] += tp
    global_metrics['fp'] += fp
    global_metrics['fn'] += fn
    global_metrics['intersection'] += intersection
    global_metrics['union'] += union


def process_class_logits(class_logits, area_threshold=100, expand_bbox=True):
    """Generate SAM prompts from coarse segmentation logits."""
    probs = torch.sigmoid(class_logits)
    binary_mask_for_cc = (probs > 0.5).detach().cpu().numpy().astype(np.uint8)
    probs_np = probs.detach().cpu().numpy().astype(np.float32)

    image_size = 1024
    batch_results = []

    for b in range(binary_mask_for_cc.shape[0]):
        current_binary = binary_mask_for_cc[b, 0, :, :]
        current_probs = probs_np[b, 0, :, :]
        sample_results = []

        num_labels, labels = cv2.connectedComponents(current_binary)
        for lab in range(1, num_labels):
            comp = labels == lab
            area = int(comp.sum())
            if area < area_threshold:
                continue

            y_idx, x_idx = np.nonzero(comp)
            min_y, max_y = int(y_idx.min()), int(y_idx.max())
            min_x, max_x = int(x_idx.min()), int(x_idx.max())

            if expand_bbox:
                bw = max_x - min_x + 1
                bh = max_y - min_y + 1
                min_x = max(0, int(min_x - 0.05 * bw))
                max_x = min(image_size - 1, int(max_x + 0.05 * bw))
                min_y = max(0, int(min_y - 0.05 * bh))
                max_y = min(image_size - 1, int(max_y + 0.05 * bh))

            focused_soft_mask = np.zeros_like(current_probs, dtype=np.float32)
            focused_soft_mask[comp] = current_probs[comp]
            soft256 = cv2.resize(focused_soft_mask, (256, 256), interpolation=cv2.INTER_LINEAR)

            comp_probs = current_probs[comp]
            argmax_flat = int(np.argmax(comp_probs))
            comp_ys, comp_xs = np.nonzero(comp)
            pos_y = int(comp_ys[argmax_flat])
            pos_x = int(comp_xs[argmax_flat])

            points = {
                "coords": np.array([[float(pos_x), float(pos_y)]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int64),
            }

            sample_results.append({
                'bbox': [min_x, min_y, max_x, max_y],
                'mask': soft256,
                'points': points,
            })

        batch_results.append(sample_results)

    return batch_results


def predict_masks_batch(
    sam_model,
    image_embeddings,
    batch_results,
    device='cuda',
    max_prompts_per_image=64,
):
    """Decode SAM masks given prompt dictionaries produced by :func:`process_class_logits`."""
    sam_model.eval()
    final_predictions = []

    for idx, sample_results in enumerate(batch_results):
        if not sample_results:
            final_predictions.append(torch.zeros((1, 1, 1024, 1024), device=device))
            continue

        if (max_prompts_per_image is not None) and (max_prompts_per_image > 0) and (len(sample_results) > max_prompts_per_image):
            chosen = np.random.choice(len(sample_results), max_prompts_per_image, replace=False)
            sample_results = [sample_results[i] for i in chosen]

        current_image_embedding = image_embeddings[idx:idx + 1]

        sparse_embeddings_list = []
        dense_embeddings_list = []

        for prompt in sample_results:
            box = torch.tensor(prompt['bbox'], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.from_numpy(prompt['mask']).float().to(device).unsqueeze(0).unsqueeze(0)

            points_arg = None
            if prompt.get('points') is not None:
                coords_np = prompt['points']['coords']
                labels_np = prompt['points']['labels']
                point_coords = torch.from_numpy(coords_np).float().to(device).unsqueeze(0)
                point_labels = torch.from_numpy(labels_np).long().to(device).unsqueeze(0)
                points_arg = (point_coords, point_labels)

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=points_arg,
                boxes=box,
                masks=mask,
            )
            sparse_embeddings_list.append(sparse_embeddings)
            dense_embeddings_list.append(dense_embeddings)

        if not sparse_embeddings_list:
            final_predictions.append(torch.zeros((1, 1, 1024, 1024), device=device))
            continue

        sparse_embeddings_all = torch.cat(sparse_embeddings_list, dim=0)
        dense_embeddings_all = torch.cat(dense_embeddings_list, dim=0)

        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=current_image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_all,
            dense_prompt_embeddings=dense_embeddings_all,
            multimask_output=False,
        )

        resized_masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode='bilinear',
            align_corners=False,
        )

        merged_mask = torch.max(resized_masks, dim=0)[0]
        final_predictions.append(merged_mask.unsqueeze(0))

    return torch.cat(final_predictions, dim=0)
