# RGR-MPLMM

**Reliability-Gated Reconstruction for Multimodal Prompt Learning with Missing Modalities**

This repository contains the official implementation of *RGR-MPLMM*, an enhancement of the MPLMM framework for multimodal sentiment analysis (MSA) under missing-modality conditions. RGR-MPLMM adds three decoupled, independently-ablatable modules on top of the MPLMM backbone:

- **M1: Modality LayerNorm (MLN)** — mitigates inter-modal scale drift before classification.
- **M2: Reliability-Gated Fusion (RGF)** — assigns per-modality reliability scores and progressively activates through zero-initialized residual injection.
- **M3: Cross-modal Reconstruction Consistency (CRC)** — supervises generative-prompt feature fidelity via a teacher-student dual forward pass (training-only, zero inference overhead).

## Highlights

- **+0.83 pp ACC / +1.15 pp F1** average improvement on CMU-MOSI under 6 missing-modality settings (5-seed mean).
- **+0.18 pp ACC / +0.13 pp F1** on CMU-MOSEI.
- **~17K additional parameters** (< 2% of the backbone) with **zero inference overhead**.
- Plug-in compatible: drops directly into any prompt-learning framework targeting missing modalities.

## Architecture

![Architecture](Figure1_architecture.png)

## Repository Structure

```
RGR-MPLMM/
├── main.py                    # Main training entry (uses src/train.py with full RGR)
├── train_ablation.py          # Ablation trainer with --use_mln/--use_rgf/--use_crc switches
├── eval_per_mode.py           # Evaluate a saved checkpoint at a fixed missing mode
├── aggregate_ablation.py      # Aggregate ablation logs → Table III rows
├── aggregate_iemocap.py       # Aggregate IEMOCAP logs (baseline vs RGR comparison)
├── src/
│   ├── model.py               # PromptModel with MLN + RGF; MULTModel for pretraining
│   ├── train.py               # Training loop with CRC dual-forward (auto-detects RGR)
│   ├── utils.py               # transfer_model (excludes MLN from freeze list)
│   ├── eval_metrics.py        # ACC / F1 / MAE / Corr / ACC-5 / ACC-7
│   ├── mosidata.py            # CMU-MOSI/MOSEI loader with missing-mode protocol
│   └── iemodata.py            # IEMOCAP loader (4-class)
├── modules/                   # MulT transformer modules
├── run_mosi_5seeds.sh         # MOSI training pipeline, 5 seeds × 6 modes
├── run_mosei_5seeds_parallel.sh   # MOSEI parallel multi-GPU
├── run_iemocap_5seeds.sh      # IEMOCAP training pipeline
└── run_mosi_ablation.sh       # MOSI single-module ablations (4 configs × 5 seeds)
```

## Requirements

- Python 3.8+
- PyTorch 1.13+ (tested with 2.1.1)
- CUDA 11.x or 12.x
- See `requirements.txt` (or install: `torch`, `scikit-learn`, `numpy`)

## Datasets

Datasets are **not included** in this repo (licensing). Obtain via the standard channels:

- **CMU-MOSI** / **CMU-MOSEI**: [MMSA toolkit](https://github.com/thuiar/MMSA) provides preprocessed `.pkl` features. Place at `./dataset/mosi_data.pkl` and `./dataset/mosei_senti_data.pkl`.
- **IEMOCAP**: [Official site](https://sail.usc.edu/iemocap/) (research license required). Place features at `./dataset/IEMOCAP_features_2021/`.

## Pretrained Checkpoint

Pre-train the MulT backbone on full-modality CMU-MOSEI (drop_rate=0):

```bash
python main.py \
    --dataset mosei \
    --data_path ./dataset/mosei_senti_data.pkl \
    --drop_rate 0 \
    --num_epochs 40 --batch_size 32 \
    --name ./pretrained/mosei.pt
```

The resulting `mosei.pt` is the shared starting point for all downstream fine-tuning.

## Reproducing Main Results

### CMU-MOSI (5 seeds × 6 missing modes)

```bash
chmod +x run_mosi_5seeds.sh
./run_mosi_5seeds.sh
```

This trains 5 seeds (111/222/333/444/555) and evaluates each checkpoint under 6 fixed missing modes (mode 0–5). Logs land in `./logs_mosi/`.

### CMU-MOSEI

```bash
chmod +x run_mosei_5seeds_parallel.sh
./run_mosei_5seeds_parallel.sh
```

### Ablation Study (MOSI)

```bash
chmod +x run_mosi_ablation.sh
./run_mosi_ablation.sh
python aggregate_ablation.py --log_dir ./logs_ablation
```

## Quick Inference Example

```python
import torch
from src.model import PromptModel

model = torch.load('./pretrained/mosi_seed111.pt')
model.eval()

# text, audio, vision are tensors of shape (B, T, D_modality)
# missing_mod: int in {0..6} indicating which modalities are missing
with torch.no_grad():
    pred = model(text, audio, vision, missing_mod=2)  # e.g. vision missing
```

## Citation

If you use RGR-MPLMM in your work, please cite:

```bibtex
@inproceedings{rgr-mplmm,
  title     = {Reliability-Gated Reconstruction for Multimodal Prompt Learning with Missing Modalities},
  author    = {Your Name and Co-authors},
  booktitle = {Proceedings of ICCSE},
  year      = {2026}
}
```

## Acknowledgments

This work builds on the [MPLMM](https://github.com/.../MPLMM) framework. We thank the authors of [MulT](https://github.com/yaohungt/Multimodal-Transformer) and the MMSA team for releasing their codebases and preprocessed features.

## License

This code is released under the MIT License. See `LICENSE` for details.
