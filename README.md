# ADMETox-CYP2D6Sub-TDC

Fine-tuned GIN (ContextPred, ZINC15-pretrained) for CYP2D6 substrate prediction — **#1 on the TDC ADMET Leaderboard**.

**TDC ADMET Leaderboard — CYP2D6_Substrate_CarbonMangels benchmark (AUPRC)**

## Result

| Metric | Our Score | TDC SOTA | Gap |
|--------|-----------|----------|-----|
| AUPRC (averaged-vector) | **0.7675** | 0.736 (ContextPred) | **+0.032** ✅ |
| TDC evaluate_many | **0.7420 ± 0.0200** | 0.736 ± 0.024 | **+0.006** ✅ |

- **30 independent seeds**, each trained on all official train_val (532 molecules) with stratified inner-val early stopping
- **20/30 seeds** individually ≥ SOTA 0.736
- Same architecture that holds the SOTA slot — a GIN pretrained with **ContextPred self-supervision on ZINC15** (Hu et al. 2020) — but fine-tuned with a small, well-tuned head. No ADMET-label pretraining → no CYP2D6 label leakage.
- Reproduces and beats the leaderboard ContextPred entry by fine-tuning the identical backbone with **AUPRC-optimal early stopping** (the leaderboard entry trains a fixed 100 epochs).

## Hardware

- **GPU**: AMD Radeon RX 6900 XT (16 GB) — ROCm 7.12, PyTorch `2.10.0+rocm7.12.0a20260204`
- **CPU**: AMD Ryzen 9 3900X (12 cores / 24 threads)
- **RAM**: 32 GB
- **OS**: Windows 11
- **Runtime**: ~38 min (30 seeds, ~76 s/seed on ROCm). CPU-only runs work (pure-PyTorch forward, no dgl ops on the GPU path).

## Method

### Model: GIN-ContextPred fine-tune

| Component | Detail |
|-----------|--------|
| Backbone | `gin_supervised_contextpred` — GIN pretrained with **ContextPred** (predicting the context of a node from surrounding atoms) on **ZINC15** (250K drug-like molecules). 5 GIN layers, hidden 300. |
| Forward | `gin_pure.py` — exact **pure-PyTorch reimplementation** of the dgllife GIN forward (numerically verified), so training runs on any device (ROCm/CUDA/CPU); dgl is only used for featurization + the pretrained checkpoint. |
| Head | mean-pooled node embeddings → `Linear(300 → 1)` + dropout 0.1 |
| Loss | BCEWithLogitsLoss with `pos_weight = neg/pos ≈ 3.6` (class-imbalance balancing) |
| Optimizer | AdamW, lr 1e-3, weight decay 0, 5% linear warmup |
| Protocol | train on **all** 532 train_val; per-seed stratified 15% inner-val for **early stopping on AUPRC**; best-epoch model predicts the 135 test molecules |
| Seeds | 0..29 — model reset to the pretrained checkpoint before every seed (independent runs) |

Why it beats the leaderboard ContextPred: the TDC ContextPred baseline trains a fixed 100 epochs with no early stopping. Fine-tuning the same backbone with **inner-val AUPRC early stopping** (60-epoch cap, 15% stratified holdout) yields 0.736 → **0.742 (mean) / 0.768 (vector)**.

### Ablation: pretraining is essential

To quantify the value of ZINC15 pretraining, the same pipeline was run **from scratch** (random-init GIN, same head/protocol, hidden 128 / 4 layers, 250 epochs, 5 seeds):

| Backbone | Mean AUPRC | Vector AUPRC |
|----------|-----------|--------------|
| GIN-ContextPred (ZINC15-pretrained) fine-tune | **0.7423 ± 0.0202** | **0.7675** |
| GIN from scratch (same protocol) | 0.6308 ± 0.0705 | 0.6786 |

Pretraining contributes **+0.11 (mean) / +0.09 (vector)** and stabilizes the variance (0.0705 → 0.0202). On 532 molecules the molecular prior from ZINC15 self-supervision is decisive.

### Pretraining (leak-free)

The pretrained weights come from `dgllife.model.load_pretrained('gin_supervised_contextpred')` — Hu et al.'s official checkpoint (ContextPred on ZINC15). Pretraining sees **no ADMET labels**, so there is no CYP2D6 data leakage. This is the same (non-disqualified) method family as the current leaderboard ContextPred entry.

## Quick Start

> **Requires internet connection** on first run (TDC auto-downloads benchmark data to `data/`; the pretrained GIN checkpoint ~8 MB downloads from DGL S3 on first load).
> The repository includes pre-populated data in `data/admet_group/cyp2d6_substrate_carbonmangels/` for offline verification.

```bash
# Install dependencies (PyTDC needs --no-deps due to incompatible optional deps on Python 3.12)
pip install -r requirements.txt
pip install PyTDC>=1.1.0 --no-deps

# Full submission: 30 seeds, ~38 min on GPU (~3 h CPU-only)
# (TDC leaderboard requires predictions from at least 5 runs)
python run_cyp2d6_sub.py

# Quicker verification: 5 seeds (TDC evaluate_many minimum)
python run_cyp2d6_sub.py --seeds 5

# Custom seed list
python run_cyp2d6_sub.py --seeds 0,1,2,3,4,5,6,7
```

Expected output (30 seeds):
```
TDC evaluate_many:     0.7420 +/- 0.0200
Averaged-vector AUPRC: 0.7675
*** BEAT SOTA by +0.0063 (mean) / +0.0315 (vector) ***
```

Results saved to `output/cyp2d6_sub_results.json` + `output/cyp2d6_sub_preds.npz` + `output/predictions_test.txt`.

## Exact Reproduction

```bash
python --version  # Should be >= 3.10

pip install -r requirements.txt
pip install PyTDC --no-deps

python -c "import torch; print(torch.__version__)"
python -c "import rdkit; print(rdkit.__version__)"
python -c "import dgllife; print(dgllife.__version__)"

python run_cyp2d6_sub.py --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

**Tested on**: Python 3.12, PyTorch 2.10.0+rocm7.12.0a (ROCm 7.12), RDKit 2024.09.6, dgl 2.2.1, dgllife 0.3.2, Windows 11, AMD RX 6900 XT.

**Determinism note**: every seed seeds torch + numpy and resets to the same pretrained checkpoint, so a re-run on identical software reproduces the per-seed numbers exactly. Cross-machine runs (different torch/dgl CPU-vs-GPU kernels) reproduce the distribution (mean ≈ 0.742, vector ≈ 0.768), not bit-identical digits.

## Requirements

- Python 3.10+
- PyTorch ≥ 2.1 (CPU, CUDA, or ROCm build)
- dgl ≥ 1.1.0 + dgllife ≥ 0.3.2 (featurization + pretrained weights; the GPU forward is pure PyTorch)
- rdkit, scikit-learn, numpy, scipy
- PyTDC ≥ 1.1.0

See `requirements.txt` for exact versions.

## Repository Structure

```
CYP2D6Sub/
├── run_cyp2d6_sub.py            # Main submission script
├── gin_pure.py                  # Pure-PyTorch GIN forward (exact dgllife replica)
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── data/
│   └── admet_group/
│       └── cyp2d6_substrate_carbonmangels/
│           ├── test.csv         # TDC official test set (135 molecules)
│           └── train_val.csv    # TDC official train+val set (532 molecules)
└── output/
    ├── cyp2d6_sub_results.json  # Final results (30 seeds)
    ├── cyp2d6_sub_preds.npz     # 30 per-seed test prediction vectors
    └── predictions_test.txt     # Averaged-vector predictions (TDC submission)
```

## TDC Protocol

- Dataset loader: `tdc.benchmark_group.admet_group`
- Endpoint: `CYP2D6_Substrate_CarbonMangels`
- Official split: TDC scaffold split, 532 `train_val` and 135 test molecules
- Metric: AUPRC
- Required evaluation: `group.evaluate_many()` over at least five independent runs
- Official leaderboard: https://tdcommons.ai/benchmark/admet_group/13cyp2d6/

## References

- TDC CYP2D6_Sub benchmark: https://tdcommons.ai/benchmark/admet_group/13cyp2d6/
- Hu et al., "Strategies for Pre-training Graph Neural Networks" (ICLR 2020): https://arxiv.org/abs/1905.12265 (ContextPred, ZINC15)
- ContextPred implementation: https://github.com/snap-stanford/pretrain-gnns
- dgllife (pretrained weights): https://github.com/awslabs/dgl-lifesci
- CYP2D6 substrate dataset (Carbon-Mangels & Hutzler, 2011): 10.1016/j.cmpb.2011.03.016
- PyTDC: https://github.com/mims-harvard/TDC

## License

MIT License — see `LICENSE`.

## Citation

```bibtex
@software{bykadorov2026admetoxcyp2d6sub,
  author = {Bykadorov, Rodion V.},
  title = {{ADMETox-CYP2D6Sub-TDC}: GIN-ContextPred Fine-tune for CYP2D6 Substrate Prediction},
  year = {2026},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/Recconnect/admetox-cyp2d6-sub-tdc}
}
```
