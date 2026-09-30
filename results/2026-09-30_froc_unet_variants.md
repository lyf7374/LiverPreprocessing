# 2026-09-30: froc unet variants

**Purpose.** FROC comparison of the plain 3D U-Net, the synthetic-small-tumour U-Net (experiment 1) and the CC-DiceCE U-Net (experiment 2) from their saved tumour probability maps, to settle whether the single argmax operating points reported so far (0.508 recall at 1.73 FP/case vs 0.455 at 0.63 vs 0.413 at 0.50) lie on one curve or whether the modifications genuinely shift the sensitivity / false-positive trade-off, per lesion size.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: small-held-out seed 0 (train 534 / val 67 / test 439); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

Inference re-run with predict.py --save-prob (sliding window, mirroring test-time augmentation, tumour softmax probability stored as uint8 x 255) on the val (67) and test (439) subsets of the small-held-out split; one one-hour gpushort job per model. scripts/froc_by_lesion_size.py binarises the maps at 21 thresholds (0.02-0.98), applies the same rules as evaluate_by_lesion_size.py (26-connected components, lesion detected when >= 10 % of its voxels are covered, false positive = predicted component touching no lesion) and interpolates the sensitivity at 0.25, 0.5, 1, 2, 4, 8 false positives per case, overall and per size bin (5-15 mm primary; 0-5 mm exploratory). Jobs: prob 29605543 / 29605546 / 29605549, FROC 29605545 / 29605548 / 29605551 (8 CPUs, 7-15 min each). Artefacts: <run>_<subset>_froc.csv (all operating points) and <run>_<subset>_froc_summary.json.

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `unet3d` = Plain 3D U-Net 300 ep; `synth` = U-Net + synthetic small tumours; `ccdicece` = U-Net + CC-DiceCE (w = 0.5)

**Findings.**

Measured false-positive ranges over the threshold sweep (test): plain U-Net 0.13-5.9 per case, CC-DiceCE 0.20-5.2, synthetic 0.61-8.5, i.e. the synthetic model cannot be operated below about 0.6 false positives per case even at threshold 0.98, so the two low-FP columns are empty for it. At matched false-positive rates on the 5-15 mm target (840 lesions): sensitivity at 1 FP/case plain 0.288, CC-DiceCE 0.322, synthetic 0.329; at 2 FP/case 0.336 / 0.384 / 0.429; mean over the measured rates 0.283 / 0.328 / 0.454. In the 5-10 mm bin: at 1 FP/case 0.103 / 0.144 / 0.165, at 2 FP/case 0.151 / 0.193 / 0.267; in the 10-15 mm bin 0.529 / 0.554 / 0.543 at 1 FP/case and 0.575 / 0.631 / 0.639 at 2. For lesions > 15 mm the synthetic model is slightly below the others at 1 FP/case (0.856 vs 0.899 plain and 0.913 CC-DiceCE) and equal at 2. Interpretation: the argmax comparison overstated the synthetic gain: the plain U-Net reaches 0.329 on 5-15 mm at 1.76 FP/case (threshold 0.10), about the same as the synthetic model at 1 FP/case, and the synthetic argmax point (0.404 at 1.56 FP/case) is about 7 points above the plain curve at that rate. So both modifications shift the curve upward, but by less than the single operating points suggested: roughly +4 (CC-DiceCE) and +4 to +9 (synthetic) sensitivity points on 5-15 mm at 1-2 FP/case, with the synthetic model's advantage concentrated in the 5-10 mm bin and at higher FP rates. The synthetic model's false-positive floor (0.6 per case) is the direct motivation for a second-stage false-positive reduction. On the validation subset (large lesions only) all-size sensitivity at 1 FP/case is 0.973 (plain), 0.941 (CC-DiceCE) and 0.871 (synthetic).

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | Plain 3D U-Net 300 ep | U-Net + synthetic small tumours | U-Net + CC-DiceCE (w = 0.5) |
|---|---:|---|---:|---:|---:|
| overall | 439 | case tumour Dice (mean) | 0.709 | 0.704 | 0.703 |
|  |  | voxel Dice (pooled) | 0.851 | 0.847 | 0.847 |
|  |  | voxel recall (pooled) | 0.827 | 0.832 | 0.833 |
|  |  | voxel precision (pooled) | 0.877 | 0.863 | 0.861 |
|  |  | lesion recall = sensitivity | 0.413 | 0.508 | 0.455 |
|  |  | lesion precision | 0.797 | 0.583 | 0.774 |
|  |  | lesion F1 | 0.544 | 0.543 | 0.573 |
|  |  | false positives per case | 0.50 | 1.73 | 0.63 |
| MCT-LTDiag | 354 | case tumour Dice (mean) | 0.730 | 0.726 | 0.725 |
|  |  | voxel Dice (pooled) | 0.856 | 0.851 | 0.852 |
|  |  | voxel recall (pooled) | 0.827 | 0.830 | 0.831 |
|  |  | voxel precision (pooled) | 0.887 | 0.873 | 0.874 |
|  |  | lesion recall = sensitivity | 0.396 | 0.488 | 0.436 |
|  |  | lesion precision | 0.821 | 0.628 | 0.800 |
|  |  | lesion F1 | 0.534 | 0.549 | 0.565 |
|  |  | false positives per case | 0.47 | 1.58 | 0.60 |
| PLC-CECT | 48 | case tumour Dice (mean) | 0.537 | 0.516 | 0.517 |
|  |  | voxel Dice (pooled) | 0.843 | 0.843 | 0.839 |
|  |  | voxel recall (pooled) | 0.835 | 0.844 | 0.850 |
|  |  | voxel precision (pooled) | 0.852 | 0.842 | 0.829 |
|  |  | lesion recall = sensitivity | 0.659 | 0.805 | 0.744 |
|  |  | lesion precision | 0.635 | 0.307 | 0.616 |
|  |  | lesion F1 | 0.647 | 0.444 | 0.674 |
|  |  | false positives per case | 0.65 | 3.10 | 0.79 |
| WAW-TACE | 37 | case tumour Dice (mean) | 0.717 | 0.725 | 0.713 |
|  |  | voxel Dice (pooled) | 0.800 | 0.803 | 0.797 |
|  |  | voxel recall (pooled) | 0.810 | 0.828 | 0.826 |
|  |  | voxel precision (pooled) | 0.790 | 0.779 | 0.769 |
|  |  | lesion recall = sensitivity | 0.587 | 0.693 | 0.613 |
|  |  | lesion precision | 0.667 | 0.505 | 0.613 |
|  |  | lesion F1 | 0.624 | 0.584 | 0.613 |
|  |  | false positives per case | 0.59 | 1.38 | 0.78 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 840 | Plain 3D U-Net 300 ep | 0.226 [0.20, 0.26] | 0.638 | 0.264 | 0.116 | 108 |
| **5-15 mm** | 840 | U-Net + synthetic small tumours | 0.405 [0.37, 0.44] | 0.473 | 0.361 | 0.223 | 379 |
| **5-15 mm** | 840 | U-Net + CC-DiceCE (w = 0.5) | 0.289 [0.26, 0.32] | 0.615 | 0.336 | 0.149 | 152 |
| 0-5 mm (exploratory) | 509 | Plain 3D U-Net 300 ep | 0.065 [0.05, 0.09] | 0.337 | 0.051 | 0.001 | 65 |
| 0-5 mm (exploratory) | 509 | U-Net + synthetic small tumours | 0.124 [0.10, 0.16] | 0.157 | 0.076 | 0.011 | 338 |
| 0-5 mm (exploratory) | 509 | U-Net + CC-DiceCE (w = 0.5) | 0.083 [0.06, 0.11] | 0.365 | 0.062 | 0.001 | 73 |
| 5-10 mm | 475 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.312 | 0.058 | 0.019 | 64 |
| 5-10 mm | 475 | U-Net + synthetic small tumours | 0.242 [0.21, 0.28] | 0.288 | 0.171 | 0.116 | 284 |
| 5-10 mm | 475 | U-Net + CC-DiceCE (w = 0.5) | 0.105 [0.08, 0.14] | 0.373 | 0.097 | 0.040 | 84 |
| 10-15 mm | 365 | Plain 3D U-Net 300 ep | 0.441 [0.39, 0.49] | 0.785 | 0.322 | 0.242 | 44 |
| 10-15 mm | 365 | U-Net + synthetic small tumours | 0.616 [0.57, 0.66] | 0.703 | 0.415 | 0.363 | 95 |
| 10-15 mm | 365 | U-Net + CC-DiceCE (w = 0.5) | 0.529 [0.48, 0.58] | 0.739 | 0.404 | 0.292 | 68 |
| > 15 mm | 742 | Plain 3D U-Net 300 ep | 0.863 [0.84, 0.89] | 0.932 | 0.830 | 0.640 | 47 |
| > 15 mm | 742 | U-Net + synthetic small tumours | 0.888 [0.86, 0.91] | 0.939 | 0.834 | 0.660 | 43 |
| > 15 mm | 742 | U-Net + CC-DiceCE (w = 0.5) | 0.898 [0.87, 0.92] | 0.926 | 0.836 | 0.660 | 53 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 781 | Plain 3D U-Net 300 ep | 0.224 [0.20, 0.25] | 0.670 | 0.267 | 0.115 | 86 |
| **5-15 mm** | 781 | U-Net + synthetic small tumours | 0.392 [0.36, 0.43] | 0.503 | 0.361 | 0.217 | 302 |
| **5-15 mm** | 781 | U-Net + CC-DiceCE (w = 0.5) | 0.283 [0.25, 0.32] | 0.641 | 0.337 | 0.146 | 124 |
| 0-5 mm (exploratory) | 508 | Plain 3D U-Net 300 ep | 0.063 [0.04, 0.09] | 0.421 | 0.051 | 0.001 | 44 |
| 0-5 mm (exploratory) | 508 | U-Net + synthetic small tumours | 0.122 [0.10, 0.15] | 0.218 | 0.076 | 0.011 | 223 |
| 0-5 mm (exploratory) | 508 | U-Net + CC-DiceCE (w = 0.5) | 0.081 [0.06, 0.11] | 0.461 | 0.062 | 0.001 | 48 |
| 5-10 mm | 458 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.346 | 0.059 | 0.020 | 53 |
| 5-10 mm | 458 | U-Net + synthetic small tumours | 0.229 [0.19, 0.27] | 0.316 | 0.161 | 0.111 | 227 |
| 5-10 mm | 458 | U-Net + CC-DiceCE (w = 0.5) | 0.100 [0.08, 0.13] | 0.407 | 0.095 | 0.039 | 67 |
| 10-15 mm | 323 | Plain 3D U-Net 300 ep | 0.455 [0.40, 0.51] | 0.817 | 0.330 | 0.249 | 33 |
| 10-15 mm | 323 | U-Net + synthetic small tumours | 0.622 [0.57, 0.67] | 0.728 | 0.421 | 0.368 | 75 |
| 10-15 mm | 323 | U-Net + CC-DiceCE (w = 0.5) | 0.542 [0.49, 0.60] | 0.754 | 0.410 | 0.298 | 57 |
| > 15 mm | 645 | Plain 3D U-Net 300 ep | 0.865 [0.84, 0.89] | 0.938 | 0.830 | 0.643 | 37 |
| > 15 mm | 645 | U-Net + synthetic small tumours | 0.893 [0.87, 0.91] | 0.943 | 0.833 | 0.666 | 35 |
| > 15 mm | 645 | U-Net + CC-DiceCE (w = 0.5) | 0.902 [0.88, 0.92] | 0.937 | 0.834 | 0.665 | 39 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 32 | Plain 3D U-Net 300 ep | 0.344 [0.20, 0.52] | 0.458 | 0.276 | 0.166 | 13 |
| **5-15 mm** | 32 | U-Net + synthetic small tumours | 0.750 [0.58, 0.87] | 0.316 | 0.520 | 0.409 | 52 |
| **5-15 mm** | 32 | U-Net + CC-DiceCE (w = 0.5) | 0.562 [0.39, 0.72] | 0.462 | 0.451 | 0.269 | 21 |
| 0-5 mm (exploratory) | 1 | Plain 3D U-Net 300 ep | 1.000 [0.21, 1.00] | 0.067 | 1.000 | 0.000 | 14 |
| 0-5 mm (exploratory) | 1 | U-Net + synthetic small tumours | 1.000 [0.21, 1.00] | 0.011 | 1.000 | 0.000 | 93 |
| 0-5 mm (exploratory) | 1 | U-Net + CC-DiceCE (w = 0.5) | 1.000 [0.21, 1.00] | 0.077 | 1.000 | 0.000 | 12 |
| 5-10 mm | 9 | Plain 3D U-Net 300 ep | 0.111 [0.02, 0.44] | 0.143 | 0.087 | 0.000 | 6 |
| 5-10 mm | 9 | U-Net + synthetic small tumours | 0.889 [0.56, 0.98] | 0.174 | 0.508 | 0.382 | 38 |
| 5-10 mm | 9 | U-Net + CC-DiceCE (w = 0.5) | 0.444 [0.19, 0.73] | 0.250 | 0.222 | 0.113 | 12 |
| 10-15 mm | 23 | Plain 3D U-Net 300 ep | 0.435 [0.26, 0.63] | 0.588 | 0.306 | 0.232 | 7 |
| 10-15 mm | 23 | U-Net + synthetic small tumours | 0.696 [0.49, 0.84] | 0.533 | 0.522 | 0.420 | 14 |
| 10-15 mm | 23 | U-Net + CC-DiceCE (w = 0.5) | 0.609 [0.41, 0.78] | 0.609 | 0.486 | 0.330 | 9 |
| > 15 mm | 49 | Plain 3D U-Net 300 ep | 0.857 [0.73, 0.93] | 0.913 | 0.836 | 0.566 | 4 |
| > 15 mm | 49 | U-Net + synthetic small tumours | 0.837 [0.71, 0.91] | 0.911 | 0.845 | 0.561 | 4 |
| > 15 mm | 49 | U-Net + CC-DiceCE (w = 0.5) | 0.857 [0.73, 0.93] | 0.894 | 0.851 | 0.561 | 5 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 27 | Plain 3D U-Net 300 ep | 0.148 [0.06, 0.32] | 0.308 | 0.194 | 0.094 | 9 |
| **5-15 mm** | 27 | U-Net + synthetic small tumours | 0.370 [0.22, 0.56] | 0.286 | 0.195 | 0.183 | 25 |
| **5-15 mm** | 27 | U-Net + CC-DiceCE (w = 0.5) | 0.148 [0.06, 0.32] | 0.364 | 0.193 | 0.092 | 7 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 22 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 13 |
| 5-10 mm | 8 | Plain 3D U-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 8 | U-Net + synthetic small tumours | 0.250 [0.07, 0.59] | 0.095 | 0.114 | 0.105 | 19 |
| 5-10 mm | 8 | U-Net + CC-DiceCE (w = 0.5) | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 19 | Plain 3D U-Net 300 ep | 0.211 [0.09, 0.43] | 0.500 | 0.219 | 0.134 | 4 |
| 10-15 mm | 19 | U-Net + synthetic small tumours | 0.421 [0.23, 0.64] | 0.571 | 0.206 | 0.215 | 6 |
| 10-15 mm | 19 | U-Net + CC-DiceCE (w = 0.5) | 0.211 [0.09, 0.43] | 0.667 | 0.218 | 0.131 | 2 |
| > 15 mm | 48 | Plain 3D U-Net 300 ep | 0.833 [0.70, 0.91] | 0.870 | 0.813 | 0.675 | 6 |
| > 15 mm | 48 | U-Net + synthetic small tumours | 0.875 [0.75, 0.94] | 0.913 | 0.831 | 0.690 | 4 |
| > 15 mm | 48 | U-Net + CC-DiceCE (w = 0.5) | 0.875 [0.75, 0.94] | 0.824 | 0.829 | 0.688 | 9 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | Plain 3D U-Net 300 ep | 0.178 | 0.235 | 0.288 | 0.336 | 0.283 |
| 5-15 mm | U-Net + synthetic small tumours | n/a | n/a | 0.329 | 0.429 | 0.454 |
| 5-15 mm | U-Net + CC-DiceCE (w = 0.5) | 0.204 | 0.277 | 0.322 | 0.384 | 0.328 |
| 5-10 mm | Plain 3D U-Net 300 ep | 0.045 | 0.068 | 0.103 | 0.151 | 0.112 |
| 5-10 mm | U-Net + synthetic small tumours | n/a | n/a | 0.165 | 0.267 | 0.294 |
| 5-10 mm | U-Net + CC-DiceCE (w = 0.5) | 0.041 | 0.097 | 0.144 | 0.193 | 0.149 |
| 10-15 mm | Plain 3D U-Net 300 ep | 0.352 | 0.452 | 0.529 | 0.575 | 0.506 |
| 10-15 mm | U-Net + synthetic small tumours | n/a | n/a | 0.543 | 0.639 | 0.661 |
| 10-15 mm | U-Net + CC-DiceCE (w = 0.5) | 0.417 | 0.512 | 0.554 | 0.631 | 0.561 |
| > 15 mm | Plain 3D U-Net 300 ep | 0.820 | 0.869 | 0.899 | 0.925 | 0.891 |
| > 15 mm | U-Net + synthetic small tumours | n/a | n/a | 0.856 | 0.894 | 0.907 |
| > 15 mm | U-Net + CC-DiceCE (w = 0.5) | 0.810 | 0.891 | 0.913 | 0.933 | 0.899 |
| all sizes | Plain 3D U-Net 300 ep | 0.371 | 0.420 | 0.459 | 0.492 | 0.453 |
| all sizes | U-Net + synthetic small tumours | n/a | n/a | 0.452 | 0.522 | 0.548 |
| all sizes | U-Net + CC-DiceCE (w = 0.5) | 0.378 | 0.447 | 0.477 | 0.517 | 0.476 |
| 0-5 mm (exploratory) | Plain 3D U-Net 300 ep | 0.033 | 0.070 | 0.098 | 0.121 | 0.093 |
| 0-5 mm (exploratory) | U-Net + synthetic small tumours | n/a | n/a | 0.065 | 0.136 | 0.180 |
| 0-5 mm (exploratory) | U-Net + CC-DiceCE (w = 0.5) | 0.036 | 0.079 | 0.097 | 0.132 | 0.104 |

### Training

```
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + synthetic small tumours: {"epochs_logged": 300, "mean_epoch_seconds": 48.1, "final_val_patch_dice_liver_tumour": [0.9574, 0.8441], "best_val_patch_dice_tumour": 0.8728, "final_train_loss": 0.2705, "synthetic_lesions_per_epoch_mean": 496.0}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | Plain 3D U-Net 300 ep | U-Net + synthetic small tumours | U-Net + CC-DiceCE (w = 0.5) |
|---|---:|---|---:|---:|---:|
| overall | 67 | case tumour Dice (mean) | 0.702 | 0.652 | 0.670 |
|  |  | voxel Dice (pooled) | 0.784 | 0.786 | 0.782 |
|  |  | voxel recall (pooled) | 0.753 | 0.772 | 0.764 |
|  |  | voxel precision (pooled) | 0.818 | 0.801 | 0.802 |
|  |  | lesion recall = sensitivity | 0.959 | 0.959 | 0.932 |
|  |  | lesion precision | 0.617 | 0.314 | 0.473 |
|  |  | lesion F1 | 0.751 | 0.473 | 0.627 |
|  |  | false positives per case | 0.66 | 2.31 | 1.15 |
| MCT-LTDiag | 19 | case tumour Dice (mean) | 0.788 | 0.768 | 0.787 |
|  |  | voxel Dice (pooled) | 0.859 | 0.835 | 0.866 |
|  |  | voxel recall (pooled) | 0.851 | 0.818 | 0.883 |
|  |  | voxel precision (pooled) | 0.868 | 0.854 | 0.850 |
|  |  | lesion recall = sensitivity | 0.960 | 0.960 | 0.960 |
|  |  | lesion precision | 0.774 | 0.393 | 0.706 |
|  |  | lesion F1 | 0.857 | 0.558 | 0.814 |
|  |  | false positives per case | 0.37 | 1.95 | 0.53 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.614 | 0.534 | 0.571 |
|  |  | voxel Dice (pooled) | 0.751 | 0.760 | 0.744 |
|  |  | voxel recall (pooled) | 0.734 | 0.769 | 0.738 |
|  |  | voxel precision (pooled) | 0.770 | 0.751 | 0.751 |
|  |  | lesion recall = sensitivity | 0.926 | 0.926 | 0.889 |
|  |  | lesion precision | 0.446 | 0.202 | 0.282 |
|  |  | lesion F1 | 0.602 | 0.331 | 0.429 |
|  |  | false positives per case | 0.91 | 2.91 | 1.79 |
| WAW-TACE | 14 | case tumour Dice (mean) | 0.757 | 0.749 | 0.707 |
|  |  | voxel Dice (pooled) | 0.817 | 0.824 | 0.822 |
|  |  | voxel recall (pooled) | 0.729 | 0.745 | 0.742 |
|  |  | voxel precision (pooled) | 0.929 | 0.922 | 0.922 |
|  |  | lesion recall = sensitivity | 1.000 | 1.000 | 0.955 |
|  |  | lesion precision | 0.786 | 0.537 | 0.778 |
|  |  | lesion F1 | 0.880 | 0.698 | 0.857 |
|  |  | false positives per case | 0.43 | 1.36 | 0.43 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 18 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 71 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 26 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 10 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 54 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 18 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 22 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 21 |
| > 15 mm | 74 | Plain 3D U-Net 300 ep | 0.959 [0.89, 0.99] | 0.910 | 0.753 | 0.726 | 7 |
| > 15 mm | 74 | U-Net + synthetic small tumours | 0.959 [0.89, 0.99] | 0.899 | 0.772 | 0.705 | 8 |
| > 15 mm | 74 | U-Net + CC-DiceCE (w = 0.5) | 0.932 [0.85, 0.97] | 0.852 | 0.764 | 0.708 | 12 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 14 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 8 |
| > 15 mm | 25 | Plain 3D U-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.851 | 0.776 | 0 |
| > 15 mm | 25 | U-Net + synthetic small tumours | 0.960 [0.80, 0.99] | 0.960 | 0.818 | 0.768 | 1 |
| > 15 mm | 25 | U-Net + CC-DiceCE (w = 0.5) | 0.960 [0.80, 0.99] | 0.923 | 0.883 | 0.783 | 2 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 48 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 24 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 33 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 17 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 11 |
| > 15 mm | 27 | Plain 3D U-Net 300 ep | 0.926 [0.77, 0.98] | 0.781 | 0.734 | 0.681 | 7 |
| > 15 mm | 27 | U-Net + synthetic small tumours | 0.926 [0.77, 0.98] | 0.806 | 0.769 | 0.661 | 6 |
| > 15 mm | 27 | U-Net + CC-DiceCE (w = 0.5) | 0.889 [0.72, 0.96] | 0.727 | 0.738 | 0.648 | 9 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 10 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 7 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| > 15 mm | 22 | Plain 3D U-Net 300 ep | 1.000 [0.85, 1.00] | 1.000 | 0.729 | 0.724 | 0 |
| > 15 mm | 22 | U-Net + synthetic small tumours | 1.000 [0.85, 1.00] | 0.957 | 0.745 | 0.686 | 1 |
| > 15 mm | 22 | U-Net + CC-DiceCE (w = 0.5) | 0.955 [0.78, 0.99] | 0.955 | 0.742 | 0.697 | 1 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |
| 5-15 mm | U-Net + synthetic small tumours | n/a | n/a | n/a | n/a | n/a |
| 5-15 mm | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | n/a |
| 5-10 mm | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |
| 5-10 mm | U-Net + synthetic small tumours | n/a | n/a | n/a | n/a | n/a |
| 5-10 mm | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | n/a |
| 10-15 mm | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |
| 10-15 mm | U-Net + synthetic small tumours | n/a | n/a | n/a | n/a | n/a |
| 10-15 mm | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | n/a |
| > 15 mm | Plain 3D U-Net 300 ep | 0.936 | 0.959 | 0.973 | 0.973 | 0.964 |
| > 15 mm | U-Net + synthetic small tumours | n/a | n/a | 0.871 | 0.950 | 0.945 |
| > 15 mm | U-Net + CC-DiceCE (w = 0.5) | 0.873 | 0.919 | 0.941 | 0.967 | 0.935 |
| all sizes | Plain 3D U-Net 300 ep | 0.936 | 0.959 | 0.973 | 0.973 | 0.964 |
| all sizes | U-Net + synthetic small tumours | n/a | n/a | 0.871 | 0.950 | 0.945 |
| all sizes | U-Net + CC-DiceCE (w = 0.5) | 0.873 | 0.919 | 0.941 | 0.967 | 0.935 |
| 0-5 mm (exploratory) | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |
| 0-5 mm (exploratory) | U-Net + synthetic small tumours | n/a | n/a | n/a | n/a | n/a |
| 0-5 mm (exploratory) | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | n/a |

### Training

```
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + synthetic small tumours: {"epochs_logged": 300, "mean_epoch_seconds": 48.1, "final_val_patch_dice_liver_tumour": [0.9574, 0.8441], "best_val_patch_dice_tumour": 0.8728, "final_train_loss": 0.2705, "synthetic_lesions_per_epoch_mean": 496.0}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Artefacts

`results/2026-09-30_froc_unet_variants/`: ccdicece_config.json, ccdicece_history.json, ccdicece_jobs.json, ccdicece_test_froc.csv, ccdicece_test_froc_summary.json, ccdicece_test_per_lesion.csv, ccdicece_test_size_binned_metrics.csv, ccdicece_val_froc.csv, ccdicece_val_froc_summary.json, ccdicece_val_per_lesion.csv, ccdicece_val_size_binned_metrics.csv, synth_config.json, synth_history.json, synth_jobs.json, synth_test_froc.csv, synth_test_froc_summary.json, synth_test_per_lesion.csv, synth_test_size_binned_metrics.csv, synth_val_froc.csv, synth_val_froc_summary.json, synth_val_per_lesion.csv, synth_val_size_binned_metrics.csv, unet3d_config.json, unet3d_history.json, unet3d_test_froc.csv, unet3d_test_froc_summary.json, unet3d_test_per_lesion.csv, unet3d_test_size_binned_metrics.csv, unet3d_val_froc.csv, unet3d_val_froc_summary.json, unet3d_val_per_lesion.csv, unet3d_val_size_binned_metrics.csv

