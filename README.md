# AAFV-Net: Age-Aware Face Verification with a Bounded False-Accept Rate

AAFV-Net decides whether two face photos taken years apart show the same person, and routes uncertain pairs to manual review instead of guessing. It is a **three-way decision framework** (auto-accept / manual review / auto-reject) whose accept threshold is frozen on a calibration split so that the false-accept rate on different people is *measured and bounded*, not assumed. It is not a new face matcher: it is built on ArcFace and other scores, and it does not claim to beat ArcFace on overall accuracy.

Status: research code and results, two test looks used (see `results/q/logs/test_access.log`). 
## What it does

1. For each pair, the older and younger image are aged and de-aged to a 5-year grid with SAM, giving up to five scores (ArcFace, EfficientNet, VGG, LPIPS, SSIM) on the original pair and on the aged pairs.
2. Three small linear learners (one per evidence branch: direct, aged, fused) feed an 11-input MLP (11 to 32 to 16 to 1). Inputs also include the age gap, image quality and an aged-face reliability score. Five seeds are averaged, then Platt-calibrated.
3. A guard layer (quality, direct-branch probability, aged-face reliability) can only turn an accept into a review, never the reverse.
4. The accept threshold is the highest probability that any different-person calibration pair reaches while passing the guards. The reject threshold is the beta-quantile of same-person calibration probabilities, so the false-reject target is set directly.
5. Zero false accepts on N person-disjoint different-person test pairs bounds the false-accept rate by 1 - alpha^(1/N). With N = 1,149 this is 0.26% (alpha = 0.05, one look) or 0.32% (alpha = 0.025, two looks).

## Results (MORPH test split, 1% false-reject target, thresholds frozen before the test)

| Method | Same-person auto-accept | False accepts: random (1,149) / all-random (277,822) / D_hard (35,735) | D_hard rejected | Neutral hard set rejected |
|---|---|---|---|---|
| **AAFV-Net** | 70.1% (66.5-73.1) | 0 / 0 / 5 | 98.0% | 57.5% |
| ArcFace only | 89.4% (88.0-90.7) | 0 / 0 / 0 | 0% | 28.5% |
| Direct-only MLP (no aged faces) | 0% | 0 / 0 / 0 | 97.8% | 50.0% |
| Siamese ResNet-18 | 0.2% | 0 / 0 / 0 | 9.9% | n/a |
| Siamese ViT-B/16 | 0% | 0 / 0 / 0 | 10.0% | n/a |
| Qwen2-VL-2B + LoRA (Unsloth), calibration subset | 0% | 0 | 10.2% | n/a |

D_hard was mined with ArcFace similarity, so it is adversarial to ArcFace by construction; the neutral hard set is mined with the Siamese models instead. Confidence intervals resample people.

External check on FG-NET (never used for training or thresholds; MORPH thresholds applied unchanged):

| Method | Adults (both >= 16 years), same-person auto-accept | Gap 30+ years | False accepts / impostors |
|---|---|---|---|
| **AAFV-Net** | 51.9% (44.5-58.4) | 10.3% | 0 / 3,670 |
| ArcFace only | 40.4% (34.5-47.4) | 3.8% | 0 / 3,670 |

On all ages, AAFV-Net made 4 false accepts in 20,000 impostor pairs, all involving children aged 0-4 (outside the training range of 16-77).



## Repository layout

| Path | Contents |
|---|---|
| `src/aafv/` | Library: pairs, gap grid, alignment, quality, pair vectors, fusion model, calibration, thresholds, duplicate-ID quarantine |
| `scripts/` | Pipeline phases `p1_` to `p10_`, baselines `p11_`, extended evaluation `p12_`, figures, release helper |
| `configs/params.yaml` | All frozen parameters |
| `tests/` | Unit tests and a synthetic-data smoke test |
| `docs/` | Data spec, pipeline spec (pseudocode per phase), evaluation and claims, file contracts, sources and licences, results notes |
| `data/morph_person_split_v5.csv` | The person-level split (file names and IDs only, no images) |
| `results/` | Frozen thresholds and their SHA-256, test-look log, ablation and extended tables, figures |


