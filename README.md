# AAFV-Net

**Age-Aware Face Verification Network with a bounded false-accept rate**

AAFV-Net is a three-way decision framework for verifying whether two face photographs taken years apart show the same person. Each pair is routed to **Auto-Accept**, **Manual Review** or **Auto-Reject**. The accept threshold is frozen on a calibration split, so the false-accept rate on different-person pairs is measured and bounded on held-out data.

## Contents

- [Method](#method)
- [Results](#results)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

## Method

1. **Evidence.** For each pair, both images are also aged or de-aged to a 5-year age grid with SAM. Five scores (ArcFace, EfficientNet, VGG, LPIPS, SSIM) are computed on the original pair and on the aged pairs.
2. **Fusion.** One linear learner per evidence branch (direct, aged, shared targets) feeds an MLP (11 inputs, hidden sizes 32 and 16, one output). Additional inputs: age gap, image quality, aged-face reliability, branch availability. Five seeds are averaged and Platt-calibrated.
3. **Guards.** Image quality, direct-branch probability and aged-face reliability can only change an accept to a review.
4. **Thresholds.** The accept threshold is the highest calibrated probability reached by any different-person calibration pair that passes the guards. The reject threshold is the beta-quantile of same-person calibration probabilities (beta = false-reject target).
5. **Bound.** With zero false accepts on N person-disjoint different-person test pairs, the false-accept rate is at most 1 - alpha^(1/N): 0.26% for N = 1,149 at alpha = 0.05, or 0.32% at alpha = 0.025 (two test looks).

## Results

**MORPH test split** (person-disjoint; thresholds frozen before testing; false-reject target 1%).

| Method | Same-person auto-accept | False accepts: D_random (1,149) | D_all_random (277,822) | D_hard (35,735) | D_hard rejected | Neutral set rejected |
|---|---|---|---|---|---|---|
| **AAFV-Net** | 70.1% (66.5-73.1) | 0 | 0 | 5 | 98.0% | 57.5% |
| ArcFace only | 89.4% (88.0-90.7) | 0 | 0 | 0 | 0% | 28.5% |
| Direct-only MLP | 0% | 0 | 0 | 0 | 97.8% | 50.0% |
| Siamese ResNet-18 | 0.2% | 0 | 0 | 0 | 9.9% | n/a |
| Siamese ViT-B/16 | 0% | 0 | 0 | 0 | 10.0% | n/a |
| Qwen2-VL-2B + LoRA (calibration subset) | 0% | 0 | n/a | 0 | 10.2% | n/a |

95% intervals resample people. D_hard is mined with ArcFace similarity. The neutral set is the top 1% of random impostor pairs ranked by the Siamese models; a model is not scored on a set it mined.

**FG-NET external evaluation** (MORPH thresholds applied unchanged; adults: both ages >= 16).

| Method | Same-person auto-accept | Gap 30+ years | False accepts |
|---|---|---|---|
| **AAFV-Net** | 51.9% (44.5-58.4) | 10.3% | 0 / 3,670 |
| ArcFace only | 40.4% (34.5-47.4) | 3.8% | 0 / 3,670 |

On all ages, AAFV-Net accepts 29.9% (ArcFace only: 21.3%) and makes 4 false accepts in 20,000 impostor pairs, all involving children aged 0-4.

## Repository Structure

```
src/aafv/            library: pairs, gap grid, alignment, quality, fusion, calibration, thresholds
scripts/             pipeline phases p1-p10, baselines p11, extended evaluation p12, release helper
configs/params.yaml  frozen parameters
tests/               unit and synthetic smoke tests
docs/                data, pipeline, evaluation and licence documentation
data/                person-level split (identifiers only)
results/             frozen thresholds, test-access log, result tables, figures
```

## Installation

```bash
git clone <repository-url> && cd <repository>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.13 (CUDA 13) and an NVIDIA Blackwell GPU. The VLM baseline requires a separate environment with `unsloth`.


## Usage

Run the phases in order; each script documents its arguments in its docstring.

| Phase | Script |
|---|---|
| Align and measure quality | `scripts/p1_align.py` |
| Age estimates | `scripts/p2_features.py` |
| Build pairs | `scripts/p3_build_pairs.py` |
| Generate aged faces | `scripts/p4_sam.py` |
| Face features, pair scores, pair vectors | `scripts/p5_faces.py`, `p5_scores.py`, `p5_vectors.py` |
| Train fusion model | `scripts/p6_train.py` |
| Quarantine duplicate identities | `scripts/p_quarantine.py` |
| Calibrate and freeze thresholds | `scripts/p78_calibrate_freeze.py --freeze` |
| Test (uses one look) | `scripts/p9_test.py --i_understand_this_uses_a_look` |
| Ablations | `scripts/p10_ablations.py`, `p10_run_all.py` |
| Baselines | `scripts/p11_siamese.py`, `p11_vlm.py`, `p11_register.py` |
| Extended evaluation (neutral sets, intervals, FG-NET) | `scripts/p12_extended.py` |
| Figures | `scripts/p11_figures.py` |

Tests: `PYTHONPATH=src pytest tests/`.

## Reproducibility

- All random seeds and parameters are in `configs/params.yaml`.
- Frozen thresholds: `results/q/thresholds/thresholds.json`, SHA-256 `4b7af9772d209ce6315d36f36df6b191728c830159f04355e7a2238fd9732cfe`.
- The test split was opened twice; see `results/q/logs/test_access.log`.

## Citation

```bibtex
@misc{aafvnet2026,
  title  = {AAFV-Net: Age-Aware Face Verification with a Bounded False-Accept Rate},
  author = {Nayak, Anurag and Hira, Swati and others},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

