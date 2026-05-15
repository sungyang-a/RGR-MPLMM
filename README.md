# RGR-MPLMM

Official implementation of:

> **Reliability-Gated Reconstruction for Multimodal Prompt Learning with Missing Modalities**

RGR-MPLMM is a lightweight enhancement framework for multimodal prompt learning under missing-modality settings.  
The method is built on top of MPLMM and introduces three additional modules:

- Modality LayerNorm (MLN)
- Reliability-Gated Fusion (RGF)
- Cross-modal Reconstruction Consistency (CRC)

The framework is designed for multimodal sentiment analysis (MSA) when one or more modalities are unavailable during inference.

---

# Overview

Missing modalities are common in real-world multimodal systems due to:

- sensor failure
- noisy environments
- signal corruption
- privacy constraints

Existing prompt-learning methods can reconstruct missing modalities, but they usually treat generated features and real observed features equally during fusion.

RGR-MPLMM introduces reliability-aware fusion and reconstruction supervision to improve robustness under severe missing-modality conditions.

---

# Architecture

The proposed framework is built upon the MPLMM backbone and adds:

1. **MLN**  
   Modality-specific LayerNorm for feature-scale alignment.

2. **RGF**  
   Reliability-aware gated fusion for adaptive modality weighting.

3. **CRC**  
   Training-only reconstruction consistency supervision.

---

# Project Structure

```text
src/
├── model.py              # Model architecture
├── train.py              # Training pipeline
├── mosidata.py           # CMU-MOSI dataloader
├── moseidata.py           # CMU-MOESI dataloader
├── iemodata.py           # IEMOCAP dataloader(more)
├── eval_metrics.py       # Evaluation metrics
└── utils.py              # Utility functions
```

---

# Environment

Recommended environment:

```bash
Python 3.9
PyTorch 2.0+
CUDA 11.8
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Datasets

The experiments are conducted on:

- CMU-MOSI
- CMU-MOSEI

Please prepare the datasets following the preprocessing pipeline of MPLMM.

Dataset files should be organized as:

```text
data/
├── MOSI/
├── MOSEI/
├── IEMOCAP/
```

---

# Training

Example:

```bash
python train.py --dataset mosi
```

For missing-modality training:

```bash
python train.py --dataset mosi --drop_rate 0.7
```


# Citation

If you find this work useful, please consider citing:

```bibtex
@article{rgrmplmm2026,
  title={Reliability-Gated Reconstruction for Multimodal Prompt Learning with Missing Modalities},
  author={Shangziyang},
  journal={ICCSE},
  year={2026}
}
```

---

# Acknowledgement

This project is implemented based on the MPLMM framework and MulT backbone.

We thank the authors of previous open-source multimodal learning projects.

---

# License

This project is released under the MIT License.
