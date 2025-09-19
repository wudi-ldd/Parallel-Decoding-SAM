import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import CONFIG, image_dir, mask_dir, val_txt
from datasets import SegmentationDataset, read_split_files
from lora import LoRA_sam
from model import SegmentationHead
from segment_anything import sam_model_registry
from utils import accumulate_metrics, initialize_metrics, predict_masks_batch, process_class_logits


def _ensure_dir(path: str) -> str:
    """Create ``path`` if missing and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def _choose_interp_for_image(src_hw, dst_hw) -> int:
    """Pick interpolation mode based on whether resizing downsamples or upsamples."""
    sh, sw = src_hw
    dh, dw = dst_hw
    return cv2.INTER_AREA if (sh * sw > dh * dw) else cv2.INTER_LINEAR


def _save_mask_png(path, bin_mask_uint8):
    """Persist a binary mask as an 8-bit PNG file."""
    cv2.imwrite(path, bin_mask_uint8)


def _save_overlay(path, rgb_img_uint8, gt_mask_uint8, color=(255, 0, 0), alpha=0.35):
    """Overlay a mask onto the RGB image and store the visualization."""
    bgr = cv2.cvtColor(rgb_img_uint8, cv2.COLOR_RGB2BGR)
    color_map = np.zeros_like(bgr, dtype=np.uint8)
    color_map[..., 2] = (gt_mask_uint8 > 0).astype(np.uint8) * color[0]
    color_map[..., 1] = (gt_mask_uint8 > 0).astype(np.uint8) * color[1]
    color_map[..., 0] = (gt_mask_uint8 > 0).astype(np.uint8) * color[2]
    over = cv2.addWeighted(bgr, 1.0, color_map, alpha, 0.0)
    cv2.imwrite(path, over)


def build_models_and_load():
    """Instantiate SAM + P-SAM heads and restore persisted weights."""
    device = CONFIG['device']

    sam_model = sam_model_registry[CONFIG['model_type']](checkpoint=CONFIG['checkpoint'])
    sam_model.to(device)

    lora_sam_model = LoRA_sam(
        sam_model,
        rank=CONFIG['rank'],
        lora_dropout=CONFIG['lora_dropout'],
        lora_alpha=CONFIG['lora_alpha'],
    ).to(device)

    seg_head = SegmentationHead(
        in_channels=256,
        intermediate_channels=64,
        out_channels=CONFIG['num_classes'],
        align_corners=False,
    ).to(device)

    save_dir = os.path.join(CONFIG['save_dir_base'], CONFIG['dataset_name'])
    lora_path = os.path.join(save_dir, f"{CONFIG['save_prefix']}_lora.safetensors")
    head_path = os.path.join(save_dir, f"{CONFIG['save_prefix']}.pth")

    if os.path.isfile(lora_path):
        lora_sam_model.load_lora_parameters(lora_path)
        print(f"[OK] Loaded LoRA weights: {lora_path}")
    else:
        print(f"[WARN] LoRA weights not found: {lora_path} (eval will run with zero-initialized LoRA)")

    if os.path.isfile(head_path):
        ckpt = torch.load(head_path, map_location=device)
        seg_head.load_state_dict(ckpt, strict=True)
        print(f"[OK] Loaded head weights: {head_path}")
    else:
        print(f"[WARN] Head weights not found: {head_path}")

    lora_sam_model.eval()
    seg_head.eval()
    return lora_sam_model, seg_head


@torch.no_grad()
def evaluate_and_dump():
    """Run deterministic evaluation and dump masks, overlays, and summary metrics."""
    device = CONFIG['device']

    lora_sam_model, seg_head = build_models_and_load()

    file_list = read_split_files(val_txt)
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        sam_model=lora_sam_model.sam,
        file_list=file_list,
        device=device,
    )

    save_root = os.path.join(CONFIG['save_dir_base'], CONFIG['dataset_name'], 'eval_outputs')
    dir_orig = _ensure_dir(os.path.join(save_root, 'original'))
    dir_pred = _ensure_dir(os.path.join(save_root, 'pred_mask'))
    dir_ovgt = _ensure_dir(os.path.join(save_root, 'overlay_gt'))

    metrics = initialize_metrics()

    for idx in tqdm(range(len(dataset)), desc='[Eval]'):
        img_tensor, gt_mask_tensor = dataset[idx]
        img_batch = img_tensor.unsqueeze(0).to(device)
        gt_mask = gt_mask_tensor.unsqueeze(0).unsqueeze(0)

        image_embedding = lora_sam_model.sam.image_encoder(img_batch)
        seg_head_logits = seg_head(image_embedding)

        prompts = process_class_logits(seg_head_logits)
        final_logits = predict_masks_batch(
            lora_sam_model.sam,
            image_embedding,
            prompts,
            device=device,
        )

        pred_prob = torch.sigmoid(final_logits).cpu().numpy()
        gt_prob = gt_mask.cpu().numpy()

        accumulate_metrics(pred_prob[0, 0], gt_prob[0, 0], metrics)

        fname = dataset.image_files[idx]
        stem = os.path.splitext(fname)[0]

        img_path = os.path.join(image_dir, fname)
        orig_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if orig_bgr is None:
            orig_rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            h, w = orig_bgr.shape[:2]
            interp = _choose_interp_for_image((h, w), (1024, 1024))
            orig_bgr = cv2.resize(orig_bgr, (1024, 1024), interpolation=interp)
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dir_orig, f"{stem}_orig.png"), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))

        pred_bin = (pred_prob[0, 0] > 0.5).astype(np.uint8) * 255
        _save_mask_png(os.path.join(dir_pred, f"{stem}_pred.png"), pred_bin)

        gt_bin = (gt_prob[0, 0] > 0.5).astype(np.uint8) * 255
        _save_overlay(os.path.join(dir_ovgt, f"{stem}_overlay_gt.png"), orig_rgb, gt_bin, color=(255, 0, 0), alpha=0.35)

    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    inter = metrics['intersection']
    union = metrics['union']
    eps = 1e-6
    iou = inter / (union + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    summary = {
        'IoU': float(iou),
        'F1': float(f1),
        'Precision': float(precision),
        'Recall': float(recall),
        'TotalImages': len(dataset),
    }

    print("\n===== Test Summary =====")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(save_root, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    evaluate_and_dump()
