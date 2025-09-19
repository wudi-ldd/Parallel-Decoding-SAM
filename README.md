# P-SAM: Parallel-Decoding SAM for Domain-Driven Prompting and Robust Pore Segmentation
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Dataset & Weights Availability:** To be released.

[Paper] P-SAM: Parallel Semantic Decoding of SAM for Domain-Driven Prompt Generation in Pore Segmentation (<placeholder>) · [Authors] Dongsheng Li, Huijie Zhang, Qiushi Xia · [Affiliation] School of Information Science and Technology, Northeast Normal University, Changchun, China

## TL;DR
P-SAM introduces a parallel prompt decoder that couples sparse prompts with a lightweight segmentation head for pore segmentation and beyond. The framework prioritises parameter-efficient SAM adaptation via LoRA fine-tuning, leverages consistency constraints across prompt and mask predictions, and delivers strong transfer to multi-polarization microscopy and remote sensing regimes. Initial experiments yield promising IoU / F1 on pore datasets and set the stage for cross-domain generalisation to urban-village identification.

## Highlights
- **Parameter-efficient** SAM adaptation with a lightweight prompt decoder, end-to-end training.
- **Consistency constraints** coupling prompt generation and mask prediction (geometric + semantic).
- **Strong transfer** from pore segmentation to remote sensing (urban-village), minimal data regime.
- Clean, reproducible code with LoRA wrappers and standardised eval/export pipeline.

## News
- <YYYY-MM-DD>: Initial release (code + README).
- <YYYY-MM-DD>: Add pretrained weights and logs for <dataset>.

## Repo Structure
- `config.py`: Central experiment configuration, including dataset_name and SAM checkpoint.
- `train.py`: LoRA fine-tuning loop integrating prompt decoding and segmentation head.
- `eval.py`: Deterministic evaluation and export to `logs/<dataset_name>/eval_outputs/`.
- `datasets.py`: SAM-aware dataset wrapper for pore segmentation data splits (train/test).
- `model.py`: Parallel decoding head with multi-dilation depthwise separable branches.
- `lora.py`: Adapters injecting LoRA into SAM image encoder attention blocks.
- `utils.py`: Losses, metrics, and prompt orchestration shared by training and evaluation.
- `weights/`: Placeholder for SAM checkpoint `sam_vit_l_0b3195.pth` and LoRA safetensors.
- `datasets/`: Expected structure `datasets/<dataset_name>/{images,masks,train.txt,test.txt}`.
- `logs/`: Training traces, checkpoints, and evaluation artefacts per dataset.

## Installation
```bash
conda create -n psam python=3.10 -y
conda activate psam
pip install -r requirements.txt
# If SAM2 or custom CUDA ops are required, document the extra steps here.
```

## Quick Start
```bash
# 1. Configure dataset_name/checkpoint in config.py
# 2. Verify datasets/<dataset_name>/ splits and weights/sam_vit_l_0b3195.pth
python train.py          # Full LoRA fine-tuning
python eval.py           # Deterministic evaluation + exports
```
- Use `train.ipynb` / `eval.ipynb` for interactive debugging (clear outputs before commit).
- Logs and checkpoints are emitted to `logs/<dataset_name>/` (metric trace + safetensors).

## Datasets
- Default pore dataset follows `datasets/pore/images` and `datasets/pore/masks` with PNG files.
- Data splits (train/test) are enumerated via newline-separated lists under `datasets/<dataset_name>/`.
- For additional domains (e.g., urban-village identification), mirror the same directory schema and update `CONFIG['dataset_name']`.

## Checkpoints
- Place the official SAM backbone under `weights/sam_vit_l_0b3195.pth` (ViT-L variant).
- Fine-tuned heads are stored as `logs/<dataset_name>/best_model.pth` and LoRA adapters as `best_model_lora.safetensors`.
- Planned public weights: <placeholder> (URL / DOI pending release).

## Training
- `python train.py` performs LoRA fine-tuning with cosine schedule and Dice + BCE objectives.
- Default batch size 1 is tuned for 1024×1024 inputs on 24 GB GPUs; adjust `CONFIG['batch_size']` cautiously.
- For reproducibility, `set_seed(42)` is applied; override via environment variable if required.
- Monitor `logs/<dataset_name>/best_model_metrics.log` for IoU / F1 improvements.

## Evaluation
- `python eval.py` rebuilds the SAM + LoRA stack, evaluates the test split, and exports masks/overlays.
- Outputs are organised under `logs/<dataset_name>/eval_outputs/{original,pred_mask,overlay_gt}`.
- Use `metrics.json` for automated reporting; overlay assets provide mask boundary quality inspection.

## Results
| Dataset | IoU (%) | F1 (%) | Notes |
|---------|---------|--------|-------|
| <Beijing> | <xx.x> | <xx.x> | Multi-polarization (microscopy) pore segmentation |
| <Xi'an> | <xx.x> | <xx.x> | Remote sensing transfer (urban-village identification) |

## Model Zoo
- <placeholder>: checkpoint URL, log archive, and expected metrics.

## Reproducibility
- Deterministic CuDNN settings are enabled; confirm GPU drivers match CUDA toolkit used for SAM.
- The repository operates under Python 3.10 / PyTorch 2.x; pin dependencies via `pip freeze` when publishing.
- Provide dataset checksum manifests and evaluation logs with releases to support independent replication.

## FAQ
- **How do I switch datasets?** Update `CONFIG['dataset_name']`, refresh train/test split lists, and re-run training/evaluation.
- **Can I integrate additional prompts?** Extend `utils.process_class_logits` to emit the desired prompt schema while preserving sparse prompts.
- **Where do evaluation artefacts live?** Check `logs/<dataset_name>/eval_outputs/` for images, masks, overlays, and metrics.

## Citation
If you use P-SAM or build upon its human-in-the-loop workflows, please cite:
```
@misc{psam2025,
  title        = {P-SAM: Parallel Semantic Decoding of SAM for Domain-Driven Prompt Generation in Pore Segmentation},
  author       = {Li, Dongsheng and Zhang, Huijie and Xia, Qiushi},
  year         = {2025},
  note         = {Preprint}
}
```

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.
