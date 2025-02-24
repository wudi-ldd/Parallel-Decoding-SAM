import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from types import MethodType

from config import CONFIG, image_dir, mask_dir, train_txt, val_txt
from datasets import SegmentationDataset, read_split_files
from model import SegmentationHead, AuxiliaryClassifier
from utils import (
    dice_loss, compute_loss, initialize_metrics, accumulate_metrics,
    process_class_logits, predict_masks_batch
)
from segment_anything import sam_model_registry
from lora import LoRA_sam

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set up logging and save directories
log_dir = os.path.join(CONFIG['log_dir_base'], CONFIG['dataset_name'])
save_dir = os.path.join(CONFIG['save_dir_base'], CONFIG['dataset_name'])
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, CONFIG['log_file']),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define forward method with intermediate features
def forward_inter(self, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    inter_features = []
    for blk in self.blocks:
        x = blk(x)
        inter_features.append(x)

    x = self.neck(x.permute(0, 3, 1, 2))
    return x, inter_features

# Load SAM model and initialize LoRA
sam_model = sam_model_registry[CONFIG['model_type']](checkpoint=CONFIG['checkpoint'])
sam_model.image_encoder.forward_inter = MethodType(forward_inter, sam_model.image_encoder)
sam_model.to(CONFIG['device'])

lora_sam_model = LoRA_sam(sam_model, rank=CONFIG['rank'])
lora_sam_model.to(CONFIG['device'])

# Read train and validation file lists
train_files = read_split_files(train_txt)
val_files = read_split_files(val_txt)

# Create datasets and data loaders
train_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    sam_model=sam_model,
    file_list=train_files,
    device=CONFIG['device']
)

val_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    sam_model=sam_model,
    file_list=val_files,
    device=CONFIG['device']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False
)

# Initialize segmentation head model
model = SegmentationHead(
    in_channels=256,
    intermediate_channels=256,
    out_channels=CONFIG['num_classes'],
    align_corners=False
)
model.to(CONFIG['device'])

# Initialize auxiliary classifiers
selected_aux_layers = [5, 11, 17, 23]
aux_classifiers = nn.ModuleList([
    AuxiliaryClassifier(in_channels=1024, num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    for _ in selected_aux_layers
])

# Set parameter trainability
for param in model.parameters():
    param.requires_grad = True

for aux in aux_classifiers:
    for param in aux.parameters():
        param.requires_grad = True

for param in lora_sam_model.sam.parameters():
    param.requires_grad = False

for layer in lora_sam_model.A_weights + lora_sam_model.B_weights:
    for param in layer.parameters():
        param.requires_grad = True

# Collect all trainable parameters
lora_trainable_params = list(filter(lambda p: p.requires_grad, lora_sam_model.parameters()))
model_trainable_params = list(model.parameters())
aux_trainable_params = list(aux_classifiers.parameters())
trainable_params = lora_trainable_params + model_trainable_params + aux_trainable_params

# Initialize optimizer
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=CONFIG['learning_rate'],
    betas=CONFIG['betas'],
    weight_decay=CONFIG['weight_decay']
)

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

num_epochs = CONFIG['num_epochs']
best_composite_score = float('-inf')
best_epoch = 0

weights = CONFIG['metric_weights']
num_classes = CONFIG['num_classes']
AUX_WEIGHT = CONFIG['aux_weight']

warmup_epochs = 3
min_lr_factor = 0.01

# Learning rate scheduler
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float((epoch + 1) / warmup_epochs)
    else:
        cosine_decay = 0.5 * (1 + math.cos((epoch - warmup_epochs) * math.pi / (num_epochs - warmup_epochs)))
        return float(min_lr_factor + (1 - min_lr_factor) * cosine_decay)

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Training loop
for epoch in range(num_epochs):
    lora_sam_model.train()
    model.train()
    aux_classifiers.train()

    total_loss = 0
    num_batches = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
        images = images.to(CONFIG['device'])
        masks = masks.to(CONFIG['device']).unsqueeze(1)

        if images.dim() != 4 or masks.dim() != 4:
            logging.error(f"Invalid input dimensions: images {images.shape}, masks {masks.shape}")
            continue

        # Forward pass
        image_embedding, inter_features = lora_sam_model.sam.image_encoder.forward_inter(images)
        seg_head_logits = model(image_embedding, inter_features)

        prompts = process_class_logits(seg_head_logits)
        with torch.no_grad():
            final_predictions = predict_masks_batch(lora_sam_model.sam, image_embedding, prompts, CONFIG['device'])

        # Compute loss
        loss, seg_loss_val, sam_loss_val, aux_loss_val = compute_loss(
            seg_head_logits=seg_head_logits,
            final_predictions=final_predictions,
            masks=masks,
            loss_fn=loss_fn,
            aux_classifiers=aux_classifiers,
            inter_features=inter_features,
            selected_aux_layers=selected_aux_layers,
            aux_weight=AUX_WEIGHT,
            seg_weight=1.0,
            sam_weight=1.0,
            loss_weights=[1,1]
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

    # Validation phase
    lora_sam_model.eval()
    model.eval()
    aux_classifiers.eval()

    val_loss = 0
    num_val_batches = 0
    global_metrics_val = initialize_metrics()       # Metrics for final predictions
    global_metrics_val_seg = initialize_metrics()   # Metrics for segmentation head output

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device']).unsqueeze(1)

            if images.dim() != 4 or masks.dim() != 4:
                logging.error(f"Invalid input dimensions: images {images.shape}, masks {masks.shape}")
                continue

            # Forward pass
            image_embedding, inter_features = lora_sam_model.sam.image_encoder.forward_inter(images)
            seg_head_logits = model(image_embedding, inter_features)

            prompts = process_class_logits(seg_head_logits)
            final_predictions = predict_masks_batch(lora_sam_model.sam, image_embedding, prompts, CONFIG['device'])

            # Compute validation loss
            loss, seg_loss_val, sam_loss_val, _ = compute_loss(
                seg_head_logits=seg_head_logits,
                final_predictions=final_predictions,
                masks=masks,
                loss_fn=loss_fn,
                aux_classifiers=None,
                inter_features=None,
                selected_aux_layers=None,
                aux_weight=0.0,
                seg_weight=1.0,
                sam_weight=1.0,
                loss_weights=[1,1]
            )
            val_loss += loss.item()
            num_val_batches += 1

            # Compute metrics
            preds_seg = torch.sigmoid(seg_head_logits).cpu().numpy()  # Segmentation head predictions
            preds_final = torch.sigmoid(final_predictions).cpu().numpy()  # Final predictions
            masks_np = masks.cpu().numpy()

            for p_seg, p_final, m_gt in zip(preds_seg, preds_final, masks_np):
                # Metrics for segmentation head
                accumulate_metrics(p_seg[0], m_gt, global_metrics_val_seg)
                # Metrics for final predictions
                accumulate_metrics(p_final[0], m_gt, global_metrics_val)

    # Calculate metrics for segmentation head
    tp_seg = global_metrics_val_seg['tp']
    fp_seg = global_metrics_val_seg['fp']
    fn_seg = global_metrics_val_seg['fn']
    intersection_seg = global_metrics_val_seg['intersection']
    union_seg = global_metrics_val_seg['union']

    iou_seg = intersection_seg / (union_seg + 1e-6)
    precision_seg = tp_seg / (tp_seg + fp_seg + 1e-6)
    recall_seg = tp_seg / (tp_seg + fn_seg + 1e-6)
    f1_seg = (2 * precision_seg * recall_seg) / (precision_seg + recall_seg + 1e-6)

    # Calculate metrics for final predictions
    tp = global_metrics_val['tp']
    fp = global_metrics_val['fp']
    fn = global_metrics_val['fn']
    intersection = global_metrics_val['intersection']
    union = global_metrics_val['union']

    iou = intersection / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    avg_iou_seg = iou_seg
    avg_precision_seg = precision_seg
    avg_recall_seg = recall_seg
    avg_f1_seg = f1_seg

    avg_iou = iou
    avg_precision = precision
    avg_recall = recall
    avg_f1 = f1

    # Compute composite score
    composite_score = (
        avg_iou * weights['iou'] +
        avg_f1 * weights['f1'] +
        avg_precision * weights['precision'] +
        avg_recall * weights['recall']
    )

    avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0

    # Log and print results
    log_message = (
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"(SegHead) IoU: {avg_iou_seg:.4f}, F1: {avg_f1_seg:.4f}, Precision: {avg_precision_seg:.4f}, Recall: {avg_recall_seg:.4f}, "
        f"(Final) IoU: {avg_iou:.4f}, F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, "
        f"Composite Score: {composite_score:.4f}, "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )
    logging.info(log_message)
    print(log_message)

    # Save best model
    if composite_score > best_composite_score:
        best_composite_score = composite_score
        best_epoch = epoch + 1

        strategy = 'MLA'
        lora_path = os.path.join(save_dir, f"{CONFIG['save_prefix']}_lora_{strategy}.safetensors")
        checkpoint_path = os.path.join(save_dir, f"{CONFIG['save_prefix']}_{strategy}.pth")

        if hasattr(lora_sam_model, 'save_lora_parameters'):
            lora_sam_model.save_lora_parameters(lora_path)
        else:
            logging.warning("lora_sam_model does not have save_lora_parameters method.")

        torch.save(model.state_dict(), checkpoint_path)

        save_message = (
            f"Best model saved at epoch {best_epoch} with Composite Score {best_composite_score:.4f} using {strategy} strategy"
        )
        logging.info(save_message)
        print(save_message)

    # Update learning rate
    scheduler.step()

logging.info("Training completed")
print("Training completed")