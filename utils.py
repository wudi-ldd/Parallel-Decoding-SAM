import torch
import numpy as np
import cv2
import logging

# Dice loss function
def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss.mean()

# Compute total loss including segmentation, SAM, and auxiliary losses
def compute_loss(
    seg_head_logits,
    final_predictions,
    masks,
    loss_fn,
    aux_classifiers=None,
    inter_features=None,
    selected_aux_layers=None,
    aux_weight=0.4,
    seg_weight=1.0,
    sam_weight=1.0,
    loss_weights=[1, 1]
):
    # Segmentation head loss
    seg_bce = loss_fn(seg_head_logits, masks)
    seg_dice = dice_loss(seg_head_logits, masks)
    seg_main_loss = loss_weights[0] * seg_bce + loss_weights[1] * seg_dice

    # SAM prediction loss
    sam_bce = loss_fn(final_predictions, masks)
    sam_dice = dice_loss(final_predictions, masks)
    sam_main_loss = loss_weights[0] * sam_bce + loss_weights[1] * sam_dice

    # Auxiliary loss
    total_aux_loss = torch.tensor(0.0, device=seg_head_logits.device)
    if aux_classifiers is not None and inter_features is not None and selected_aux_layers is not None:
        aux_losses = []
        for idx, aux_cls in zip(selected_aux_layers, aux_classifiers):
            feature_idx = idx - 1
            if feature_idx < len(inter_features):
                aux_feat = inter_features[feature_idx]
                aux_logits = aux_cls(aux_feat)
                loss_aux = loss_fn(aux_logits, masks)
                aux_losses.append(loss_aux)
            else:
                logging.warning(f"inter_features does not have index {feature_idx}")

        if aux_losses:
            aux_loss_mean = torch.mean(torch.stack(aux_losses))
            total_aux_loss = aux_loss_mean * aux_weight

    # Total loss
    total_loss = seg_weight * seg_main_loss + sam_weight * sam_main_loss + total_aux_loss

    return (
        total_loss,
        seg_main_loss.item(),
        sam_main_loss.item(),
        total_aux_loss.item()
    )

# Initialize metrics dictionary
def initialize_metrics():
    return {
        'tp': 0,  # True positives
        'fp': 0,  # False positives
        'fn': 0,  # False negatives
        'intersection': 0,  # Intersection for IoU
        'union': 0   # Union for IoU
    }

# Accumulate metrics for evaluation
def accumulate_metrics(preds, targets, global_metrics, threshold=0.5):
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

# Process segmentation logits to generate prompts
def process_class_logits(class_logits):
    probs = torch.sigmoid(class_logits)
    binary_masks = (probs > 0.5).cpu().numpy().astype(np.uint8)  # [B,1,1024,1024]
    image_size = 1024  # Image width and height
    batch_results = []

    for batch_idx in range(binary_masks.shape[0]):
        current_mask = binary_masks[batch_idx, 0, :, :]
        num_labels, labels = cv2.connectedComponents(current_mask)
        sample_results = []

        for label in range(1, num_labels):
            current_component = (labels == label).astype(np.uint8)
            y_coords, x_coords = np.nonzero(current_component)

            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)

            # Expand bounding box slightly
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y

            expanded_min_x = max(0, int(min_x - 0.05 * bbox_width))
            expanded_max_x = min(image_size - 1, int(max_x + 0.05 * bbox_width))
            expanded_min_y = max(0, int(min_y - 0.05 * bbox_height))
            expanded_max_y = min(image_size - 1, int(max_y + 0.05 * bbox_height))

            mask = np.zeros_like(current_component, dtype=np.uint8)
            mask[min_y:max_y+1, min_x:max_x+1] = current_component[min_y:max_y+1, min_x:max_x+1]

            mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            sample_results.append({
                'bbox': [expanded_min_x, expanded_min_y, expanded_max_x, expanded_max_y],
                'mask': mask_resized
            })

        batch_results.append(sample_results)

    return batch_results

# Predict masks in batch using SAM model
def predict_masks_batch(sam_model, image_embeddings, batch_results, device='cuda'):
    sam_model.eval()
    final_predictions = []
    for idx, sample_results in enumerate(batch_results):
        if not sample_results:
            final_predictions.append(torch.zeros((1, 1, 1024, 1024), device=device))
            continue

        current_image_embedding = image_embeddings[idx:idx+1]

        sparse_embeddings_list = []
        dense_embeddings_list = []

        for mask_info in sample_results:
            box = torch.tensor(mask_info['bbox'], dtype=torch.float, device=device).unsqueeze(0)
            mask = torch.from_numpy(mask_info['mask']).float().to(device).unsqueeze(0).unsqueeze(0)

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box,
                masks=mask
            )

            sparse_embeddings_list.append(sparse_embeddings)
            dense_embeddings_list.append(dense_embeddings)

        if len(sparse_embeddings_list) == 0:
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
            align_corners=False
        )

        merged_mask = torch.max(resized_masks, dim=0)[0]  # [1,1024,1024]
        final_predictions.append(merged_mask.unsqueeze(0))  # [1,1,1024,1024]

    return torch.cat(final_predictions, dim=0)  # [B,1,1024,1024]