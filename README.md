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
wget <official_sam_vit_l_url> -O weights/sam_vit_l_0b3195.pth

# 3. Single-GPU fine-tuning and evaluation
python train.py
python eval.py
```
- Training artefacts: `logs/<dataset_name>/best_model.pth`, `best_model_lora.safetensors`, `best_model_metrics.log`.
- Evaluation results: `logs/<dataset_name>/eval_outputs/{original,pred_mask,overlay_gt}/` and `metrics.json`.

## 🧪 Reproducibility
- Deterministic setup: `set_seed(42)`, LoRA rank = 1, batch size = 1.
- Optimisation: 100 epochs, AdamW lr = 1e-4, 3-epoch linear warmup, cosine decay to 1% of the initial lr.
- Expected CTS-Pore results: IoU ≈ 84.4 ± 0.5, F1 ≈ 91.5 ± 0.3. Significant deviations often indicate prompt issues or data misalignment.
- Logs and metrics are recorded under `logs/<dataset_name>/`.

## 🧱 Model Zoo (TBD)
| Dataset | Config | Weights | Notes |
| --- | --- | --- | --- |
| CTS-Pore | rank=1, dilations {1,2,3} | _TBD_ | LoRA + head package |
| Beijing Urban Village | same | _TBD_ | Evaluation logs + visualisations |
| Xi’an Urban Village | same | _TBD_ | Evaluation logs + visualisations |

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

## 🧩 Ablations & Limitations
- Ablation (CTS-Pore) shows removing box prompts (-7.1 IoU), point prompts (-5.1 IoU), LoRA (-38.5 IoU), or SAM’s decoder (-4.1 IoU) degrades performance.
- Prompt supervision and SAM supervision jointly enforce consistency; dropping either reduces F1/IoU notably.
- Current repo targets SAM ViT-L and single-class segmentation; multi-class or larger-batch setups require additional tuning.

## ✍️ Citation
```bibtex
@inproceedings{li2024psam,
  title     = {P-SAM: Parallel Semantic Decoding of SAM for Domain-Driven Prompt Generation in Pore Segmentation},
  author    = {Li, Dongsheng and Zhang, Huijie and Xia, Qiushi},
  booktitle = {TBD},
  year      = {2024},
  note      = {Update with the final venue / DOI / arXiv identifier when available}
}
```

## 📄 License
MIT License (see `LICENSE` or add the file before releasing if not yet included).

## 🙏 Acknowledgements
- Segment Anything and SAM2 project teams for the foundational models and dataset preparation.
- LoRA community for the open implementations that inspired the parameter-efficient adaptation.
