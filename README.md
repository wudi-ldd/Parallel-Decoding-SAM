# P-SAM: Parallel Semantic Decoding of SAM for Domain-Driven Prompt Generation in Pore Segmentation
> Parameter-efficient prompting for SAM with only 0.16M tunable weights, delivering cross-domain pore segmentation.


## 🔹 Highlights
- Parallel Semantic Decoding (P-SAM) unifies prompt generation and mask prediction inside SAM’s pipeline.
- LoRA rank 1 (0.16M parameters) adapts SAM without touching backbone weights.
- Multi-modal prompts (box, point, soft mask) are jointly supervised for boundary-tight segmentation.
- Strong cross-domain transfer: CTS pores → Beijing/Xi’an urban-village datasets, achieving 72.9% / 80.0% IoU.

## 📦 Environment
```text
Python >= 3.10
CUDA >= 11.7 (optional, GPU)
PyTorch >= 2.1
Accompanying packages: torchvision, numpy, opencv-python, tqdm, safetensors, pyyaml, matplotlib
```
Install the baseline dependencies with:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
> `requirements.txt` provides a minimal set of packages; adjust versions to match your local CUDA / PyTorch configuration.

## 📂 Data
```
datasets/
  <dataset_name>/
    images/
    masks/
    train.txt
    val.txt
    test.txt
```
- `<split>.txt` files enumerate file names (with extension) relative to the `images/` folder, one per line.
- Masks share the same file names and reside in `masks/`.

## 🚀 Quick Start
```bash
# 1. Create and activate an environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download SAM ViT-L weights
mkdir -p weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \
     -O weights/sam_vit_l_0b3195.pth

# 3. Single-GPU fine-tuning and evaluation
python train.py
python eval.py
```
- Training artefacts: `logs/<dataset_name>/best_model.pth`, `best_model_lora.safetensors`, `best_model_metrics.log`.
- Evaluation results: `logs/<dataset_name>/eval_outputs/{original,pred_mask,overlay_gt}/` and `metrics.json`.

## 📊 Results
| Method | IoU (%) | F1 (%) | Params (M) |
| --- | --- | --- | --- |
| SegNeXt | 77.37 | 88.12 | 13.7 |
| SAM2-UNet | 53.72 | 64.28 | 11.4 |
| **P-SAM (ours)** | **84.39** | **91.53** | **0.16** |

| Dataset | IoU (%) | F1 (%) | Precision (%) | Recall (%) |
| --- | --- | --- | --- | --- |
| Beijing Urban Village | 72.9 | 84.4 | 84.6 | 84.1 |
| Xi’an Urban Village | 80.0 | 88.9 | 89.3 | 88.5 |

Reproduce by running `python eval.py` and inspecting `logs/<dataset_name>/eval_outputs/` for overlays and masks.

## ✍️ Citation
```bibtex

```

## 🙏 Acknowledgements
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) project teams for providing the foundational model.
- [UV-SAM](https://github.com/tsinghua-fib-lab/UV-SAM) from Tsinghua FIB Lab for releasing their SAM adaptation baseline.

