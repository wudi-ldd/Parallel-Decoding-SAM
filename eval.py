import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import logging
from segment_anything import sam_model_registry
from tqdm import tqdm
from lora import LoRA_sam  # Ensure you have a module named LoRA_sam
from types import MethodType

# Configuration dictionary placed prominently at the top
CONFIG = {
    'dataset_name': 'xian',             # Dataset name
    'data_base_dir': 'datasets',           # Base directory for datasets
    'log_dir_base': 'logs',                # Base directory for logs
    'log_file': 'eval_metrics.log',        # Log file name
    'save_dir_base': 'logs',               # Base directory for saved models
    'save_prefix': 'best_model',           # Prefix for saved model files
    'model_type': 'vit_l',                 # SAM model type (e.g., vit_l)
    'checkpoint': 'weights/sam_vit_l_0b3195.pth',  # Path to SAM pre-trained weights
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Device for computation
    'num_classes': 1,                      # Number of output classes
    'threshold': 0.5,                      # Threshold for binary segmentation
    'save_predictions': False              # Option to save predictions (set True to enable)
}

# Set random seed for reproducibility (optional)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Read file list from a split file
def read_split_files(file_path):
    with open(file_path, 'r') as f:
        file_names = f.read().strip().split('\n')
    return file_names

# Forward_inter function required by SAM
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

# Dataset class for segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, sam_model, file_list, mask_size=(1024, 1024), device='cpu'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sam_model = sam_model
        self.mask_size = mask_size
        self.device = device
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and f.replace('.png', '') in file_list]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        mask_file = image_file
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)

        input_image_torch = torch.as_tensor(image, dtype=torch.float32).to(self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()

        input_image = self.sam_model.preprocess(input_image_torch.to(self.device))
        mask = torch.as_tensor(mask, dtype=torch.float32).to(self.device)

        return input_image, mask

# Auxiliary classifier for intermediate features
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(AuxiliaryClassifier, self).__init__()
        self.aux_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.aux_bn1 = nn.BatchNorm2d(256)
        self.aux_relu1 = nn.ReLU(inplace=True)
        self.aux_conv2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.aux_conv1(x)
        x = self.aux_bn1(x)
        x = self.aux_relu1(x)
        x = self.aux_conv2(x)
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        return x

# Segmentation head with MLA strategy
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels=1, align_corners=False):
        super(SegmentationHead, self).__init__()
        self.align_corners = align_corners

        # MLA branches for multi-layer feature processing
        self.mla_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        self.mla_image_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.mla_classifier_branch = nn.Sequential(
            nn.Conv2d(256 * 5, intermediate_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, image_embedding, inter_features):
        if inter_features is None:
            raise ValueError("inter_features must be provided")
        if len(inter_features) < 24:
            raise ValueError(f"Expected at least 24 inter_features, got {len(inter_features)}")

        selected_features = [inter_features[i] for i in [5, 11, 17, 23]]
        selected_features = [feat.permute(0, 3, 1, 2) for feat in selected_features]

        processed_features = []
        for i, feat in enumerate(selected_features):
            branch = self.mla_branches[i]
            x_feat = branch(feat)
            x_feat = F.interpolate(x_feat, scale_factor=4, mode='bilinear', align_corners=self.align_corners)
            processed_features.append(x_feat)

        img_feat = self.mla_image_branch(image_embedding)
        img_feat = F.interpolate(img_feat, scale_factor=4, mode='bilinear', align_corners=self.align_corners)
        processed_features.append(img_feat)

        aggregated = torch.cat(processed_features, dim=1)
        x = self.mla_classifier_branch(aggregated)
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=self.align_corners)
        return x

# Process segmentation head logits into masks and bounding boxes
def process_class_logits(class_logits):
    probs = torch.sigmoid(class_logits)
    binary_masks = (probs > 0.5).cpu().numpy().astype(np.uint8)
    image_size = 1024
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

# Predict final masks using SAM's mask decoder
def predict_masks_batch(sam_model, image_embeddings, batch_results, device='cuda'):
    sam_model.eval()
    final_predictions = []
    with torch.no_grad():
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

            resized_masks = F.interpolate(
                low_res_masks,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )

            merged_mask = torch.max(resized_masks, dim=0)[0]
            final_predictions.append(merged_mask.unsqueeze(0))
    return torch.cat(final_predictions, dim=0)

# Initialize metrics dictionary
def initialize_metrics():
    return {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'intersection': 0,
        'union': 0
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

# Compute total loss including auxiliary losses
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
    seg_bce = loss_fn(seg_head_logits, masks)
    seg_dice = dice_loss(seg_head_logits, masks)
    seg_main_loss = loss_weights[0] * seg_bce + loss_weights[1] * seg_dice

    sam_bce = loss_fn(final_predictions, masks)
    sam_dice = dice_loss(final_predictions, masks)
    sam_main_loss = loss_weights[0] * sam_bce + loss_weights[1] * sam_dice

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

    total_loss = seg_weight * seg_main_loss + sam_weight * sam_main_loss + total_aux_loss

    return (
        total_loss,
        seg_main_loss.item(),
        sam_main_loss.item(),
        total_aux_loss.item()
    )

# Evaluation function
def evaluate(
    model,
    lora_sam_model,
    aux_classifiers,
    dataloader,
    device,
    threshold=0.5
):
    model.eval()
    lora_sam_model.eval()
    aux_classifiers.eval()

    global_metrics_final = initialize_metrics()  # Metrics for SAM final predictions
    global_metrics_seg = initialize_metrics()    # Metrics for SegmentationHead outputs

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            if images.dim() != 4 or masks.dim() != 4:
                logging.error(f"Invalid input dimensions: images {images.shape}, masks {masks.shape}")
                continue

            image_embedding, inter_features = lora_sam_model.sam.image_encoder.forward_inter(images)
            seg_head_logits = model(image_embedding, inter_features)

            prompts = process_class_logits(seg_head_logits)
            final_predictions = predict_masks_batch(lora_sam_model.sam, image_embedding, prompts, device)

            preds_seg = torch.sigmoid(seg_head_logits).cpu().numpy()
            preds_final = torch.sigmoid(final_predictions).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for p_seg, p_final, m_gt in zip(preds_seg, preds_final, masks_np):
                accumulate_metrics(p_seg[0], m_gt[0], global_metrics_seg, threshold=threshold)
                accumulate_metrics(p_final[0], m_gt[0], global_metrics_final, threshold=threshold)

    # Compute metrics for SegmentationHead
    tp_seg = global_metrics_seg['tp']
    fp_seg = global_metrics_seg['fp']
    fn_seg = global_metrics_seg['fn']
    intersection_seg = global_metrics_seg['intersection']
    union_seg = global_metrics_seg['union']

    iou_seg = intersection_seg / (union_seg + 1e-6)
    precision_seg = tp_seg / (tp_seg + fp_seg + 1e-6)
    recall_seg = tp_seg / (tp_seg + fn_seg + 1e-6)
    f1_seg = (2 * precision_seg * recall_seg) / (precision_seg + recall_seg + 1e-6)

    # Compute metrics for SAM final predictions
    tp = global_metrics_final['tp']
    fp = global_metrics_final['fp']
    fn = global_metrics_final['fn']
    intersection = global_metrics_final['intersection']
    union = global_metrics_final['union']

    iou = intersection / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    log_message = (
        f"SegmentationHead Metrics - IoU: {iou_seg:.4f}, Precision: {precision_seg:.4f}, "
        f"Recall: {recall_seg:.4f}, F1: {f1_seg:.4f}\n"
        f"SAM Final Prediction Metrics - IoU: {iou:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    logging.info(log_message)
    print(log_message)

    return {
        'SegmentationHead': {
            'IoU': iou_seg,
            'Precision': precision_seg,
            'Recall': recall_seg,
            'F1': f1_seg
        },
        'SAM_Final_Prediction': {
            'IoU': iou,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    }

# Main execution function
def main():
    # Construct paths based on dataset_name
    image_dir = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'images')
    mask_dir = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'masks')
    test_txt = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'test.txt')

    # Set up logging and save directories
    log_dir = os.path.join(CONFIG['log_dir_base'], CONFIG['dataset_name'])
    save_dir = os.path.join(CONFIG['save_dir_base'], CONFIG['dataset_name'])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, CONFIG['log_file']),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = CONFIG['device']

    # Load SAM model and initialize LoRA
    sam_model = sam_model_registry[CONFIG['model_type']](checkpoint=CONFIG['checkpoint'])
    sam_model.image_encoder.forward_inter = MethodType(forward_inter, sam_model.image_encoder)
    sam_model.to(device)

    r = 512  # LoRA rank
    lora_sam_model = LoRA_sam(sam_model, rank=r)
    lora_sam_model.to(device)

    # Initialize SegmentationHead
    model = SegmentationHead(
        in_channels=256,
        intermediate_channels=256,
        out_channels=CONFIG['num_classes'],
        align_corners=False
    )
    model.to(device)

    # Initialize auxiliary classifiers
    selected_aux_layers = [5, 11, 17, 23]
    aux_classifiers = nn.ModuleList([
        AuxiliaryClassifier(in_channels=1024, num_classes=CONFIG['num_classes']).to(device)
        for _ in selected_aux_layers
    ])

    # Load trained weights for SegmentationHead
    seg_head_checkpoint = os.path.join(save_dir, f"{CONFIG['save_prefix']}_MLA.pth")
    if not os.path.exists(seg_head_checkpoint):
        logging.error(f"SegmentationHead checkpoint not found at {seg_head_checkpoint}")
        print(f"SegmentationHead checkpoint not found at {seg_head_checkpoint}")
        return
    try:
        model.load_state_dict(torch.load(seg_head_checkpoint, map_location=device))
        logging.info(f"Loaded SegmentationHead weights from {seg_head_checkpoint}")
        print(f"Loaded SegmentationHead weights from {seg_head_checkpoint}")
    except Exception as e:
        logging.error(f"Error loading SegmentationHead weights: {e}")
        print(f"Error loading SegmentationHead weights: {e}")
        return

    # Load LoRA weights
    lora_checkpoint = os.path.join(save_dir, f"{CONFIG['save_prefix']}_lora_MLA.safetensors")
    if not os.path.exists(lora_checkpoint):
        logging.error(f"LoRA checkpoint not found at {lora_checkpoint}")
        print(f"LoRA checkpoint not found at {lora_checkpoint}")
        return
    try:
        lora_sam_model.load_lora_parameters(lora_checkpoint)
        lora_sam_model.to(device)
        logging.info(f"Loaded LoRA parameters from {lora_checkpoint}")
        print(f"Loaded LoRA parameters from {lora_checkpoint}")
    except Exception as e:
        logging.error(f"Error loading LoRA parameters: {e}")
        print(f"Error loading LoRA parameters: {e}")
        return

    # Prepare test dataset and dataloader
    test_files = read_split_files(test_txt)
    test_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        sam_model=sam_model,
        file_list=test_files,
        device=device
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Perform evaluation
    metrics = evaluate(
        model=model,
        lora_sam_model=lora_sam_model,
        aux_classifiers=aux_classifiers,
        dataloader=test_loader,
        device=device,
        threshold=CONFIG['threshold']
    )

    # Print final evaluation metrics
    print("\n=== Evaluation Metrics on Test Set ===")
    print("SegmentationHead:")
    for metric, value in metrics['SegmentationHead'].items():
        print(f"  {metric}: {value:.4f}")
    print("SAM Final Prediction:")
    for metric, value in metrics['SAM_Final_Prediction'].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()