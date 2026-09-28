# 2026-09-28: unet cc dicece loss

**Purpose.** Experiment 2 of the 3D U-Net modification series: does a connected-component-aware loss (CC-DiceCE, weighted 0.5 with the standard DiceCE) make the network more sensitive to small lesions, and at what cost in false positives? Compared with the two 300-epoch baselines and with experiment 1 (synthetic small tumours) on the same split.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: small-held-out seed 0 (train 534 / val 67 / test 439); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

CC-DiceCE follows CC-Metrics (Jaus et al. 2024) as used as a training loss in the BiCC paper: every training patch is partitioned into the Voronoi cells of its ground-truth tumour components (nearest component by Euclidean distance, computed in the data loader with a distance transform); Dice + cross-entropy of the tumour class is evaluated inside every cell and the cells are averaged, so each lesion counts once regardless of its volume; patches without a tumour fall back to the whole-patch DiceCE. The training loss is (1 - w) * DiceCE + w * CC-DiceCE with w = 0.5, applied at every deep-supervision level (unet3d/losses.py). No synthetic lesions.

Everything else equals the plain 3D U-Net baseline: plans-derived architecture (30,789,391 parameters), 300 epochs x 250 iterations, batch 2, patch 112 x 128 x 160, SGD 0.01 with Nesterov momentum 0.99 and polynomial decay, 33 % foreground sampling, normalisation clip [-200, 200] HU to [-1, 1], mirroring at test time; multiprocess patch loader with 6 workers (61 s per epoch, training 5.1 h, prediction + evaluation 55 min). Run: unet3d_run/ccdicece, jobs 28984086 (train) and 28984087 (predict), args --loss ccdicece --cc-weight 0.5.

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `nnunet` = nnU-Net 300 ep; `unet3d` = Plain 3D U-Net 300 ep; `synth` = U-Net + synthetic small tumours; `ccdicece` = U-Net + CC-DiceCE (w = 0.5)

**Findings.**

CC-DiceCE gives a moderate, precision-preserving gain (plain U-Net -> CC-DiceCE): lesion sensitivity 0-5 mm 0.065 -> 0.083, 5-10 mm 0.061 -> 0.105, 10-15 mm 0.441 -> 0.529, > 15 mm 0.863 -> 0.898; overall lesion recall 0.413 -> 0.455 with lesion precision 0.797 -> 0.774 and false positives per case 0.50 -> 0.63, so the lesion F1 improves from 0.544 to 0.573 (the best of the four runs). Case-level tumour Dice (0.709 vs 0.703) and pooled voxel Dice (0.851 vs 0.847) are unchanged. Compared with experiment 1, synthesis recovers more small lesions (5-10 mm 0.242 vs 0.105) but at 1.73 false positives per case versus 0.63 here. Per dataset the pattern is the same (PLC-CECT recall 0.659 -> 0.744 at 0.79 FP per case; WAW-TACE 0.587 -> 0.613 at 0.78). On the validation subset (large lesions only) recall drops slightly (0.959 -> 0.932) and false positives rise (0.66 -> 1.15 per case), i.e. the loss shifts some capacity from large to small components. The two mechanisms are complementary (loss re-weighting vs. extra small-lesion training signal); combining them, and tuning w, is the natural next step after experiment 3.

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | U-Net + synthetic small tumours | U-Net + CC-DiceCE (w = 0.5) |
|---|---:|---|---:|---:|---:|---:|
| overall | 439 | case tumour Dice (mean) | 0.713 | 0.709 | 0.704 | 0.703 |
|  |  | voxel Dice (pooled) | 0.828 | 0.851 | 0.847 | 0.847 |
|  |  | voxel recall (pooled) | 0.795 | 0.827 | 0.832 | 0.833 |
|  |  | voxel precision (pooled) | 0.865 | 0.877 | 0.863 | 0.861 |
|  |  | lesion recall = sensitivity | 0.413 | 0.413 | 0.508 | 0.455 |
|  |  | lesion precision | 0.786 | 0.797 | 0.583 | 0.774 |
|  |  | lesion F1 | 0.542 | 0.544 | 0.543 | 0.573 |
|  |  | false positives per case | 0.54 | 0.50 | 1.73 | 0.63 |
| MCT-LTDiag | 354 | case tumour Dice (mean) | 0.733 | 0.730 | 0.726 | 0.725 |
|  |  | voxel Dice (pooled) | 0.835 | 0.856 | 0.851 | 0.852 |
|  |  | voxel recall (pooled) | 0.799 | 0.827 | 0.830 | 0.831 |
|  |  | voxel precision (pooled) | 0.874 | 0.887 | 0.873 | 0.874 |
|  |  | lesion recall = sensitivity | 0.398 | 0.396 | 0.488 | 0.436 |
|  |  | lesion precision | 0.799 | 0.821 | 0.628 | 0.800 |
|  |  | lesion F1 | 0.531 | 0.534 | 0.549 | 0.565 |
|  |  | false positives per case | 0.55 | 0.47 | 1.58 | 0.60 |
| PLC-CECT | 48 | case tumour Dice (mean) | 0.540 | 0.537 | 0.516 | 0.517 |
|  |  | voxel Dice (pooled) | 0.801 | 0.843 | 0.843 | 0.839 |
|  |  | voxel recall (pooled) | 0.763 | 0.835 | 0.844 | 0.850 |
|  |  | voxel precision (pooled) | 0.843 | 0.852 | 0.842 | 0.829 |
|  |  | lesion recall = sensitivity | 0.634 | 0.659 | 0.805 | 0.744 |
|  |  | lesion precision | 0.703 | 0.635 | 0.307 | 0.616 |
|  |  | lesion F1 | 0.667 | 0.647 | 0.444 | 0.674 |
|  |  | false positives per case | 0.46 | 0.65 | 3.10 | 0.79 |
| WAW-TACE | 37 | case tumour Dice (mean) | 0.727 | 0.717 | 0.725 | 0.713 |
|  |  | voxel Dice (pooled) | 0.801 | 0.800 | 0.803 | 0.797 |
|  |  | voxel recall (pooled) | 0.807 | 0.810 | 0.828 | 0.826 |
|  |  | voxel precision (pooled) | 0.795 | 0.790 | 0.779 | 0.769 |
|  |  | lesion recall = sensitivity | 0.573 | 0.587 | 0.693 | 0.613 |
|  |  | lesion precision | 0.683 | 0.667 | 0.505 | 0.613 |
|  |  | lesion F1 | 0.623 | 0.624 | 0.584 | 0.613 |
|  |  | false positives per case | 0.54 | 0.59 | 1.38 | 0.78 |

### Per size bin

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 509 | nnU-Net 300 ep | 0.051 [0.04, 0.07] | 0.252 | 0.048 | 0.000 | 77 |
| 0-5 mm | 509 | Plain 3D U-Net 300 ep | 0.065 [0.05, 0.09] | 0.337 | 0.051 | 0.001 | 65 |
| 0-5 mm | 509 | U-Net + synthetic small tumours | 0.124 [0.10, 0.16] | 0.157 | 0.076 | 0.011 | 338 |
| 0-5 mm | 509 | U-Net + CC-DiceCE (w = 0.5) | 0.083 [0.06, 0.11] | 0.365 | 0.062 | 0.001 | 73 |
| 5-10 mm | 475 | nnU-Net 300 ep | 0.078 [0.06, 0.11] | 0.366 | 0.066 | 0.027 | 64 |
| 5-10 mm | 475 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.312 | 0.058 | 0.019 | 64 |
| 5-10 mm | 475 | U-Net + synthetic small tumours | 0.242 [0.21, 0.28] | 0.288 | 0.171 | 0.116 | 284 |
| 5-10 mm | 475 | U-Net + CC-DiceCE (w = 0.5) | 0.105 [0.08, 0.14] | 0.373 | 0.097 | 0.040 | 84 |
| 10-15 mm | 365 | nnU-Net 300 ep | 0.425 [0.37, 0.48] | 0.791 | 0.314 | 0.237 | 41 |
| 10-15 mm | 365 | Plain 3D U-Net 300 ep | 0.441 [0.39, 0.49] | 0.785 | 0.322 | 0.242 | 44 |
| 10-15 mm | 365 | U-Net + synthetic small tumours | 0.616 [0.57, 0.66] | 0.703 | 0.415 | 0.363 | 95 |
| 10-15 mm | 365 | U-Net + CC-DiceCE (w = 0.5) | 0.529 [0.48, 0.58] | 0.739 | 0.404 | 0.292 | 68 |
| > 15 mm | 742 | nnU-Net 300 ep | 0.871 [0.84, 0.89] | 0.924 | 0.797 | 0.646 | 53 |
| > 15 mm | 742 | Plain 3D U-Net 300 ep | 0.863 [0.84, 0.89] | 0.932 | 0.830 | 0.640 | 47 |
| > 15 mm | 742 | U-Net + synthetic small tumours | 0.888 [0.86, 0.91] | 0.939 | 0.834 | 0.660 | 43 |
| > 15 mm | 742 | U-Net + CC-DiceCE (w = 0.5) | 0.898 [0.87, 0.92] | 0.926 | 0.836 | 0.660 | 53 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 508 | nnU-Net 300 ep | 0.049 [0.03, 0.07] | 0.309 | 0.048 | 0.000 | 56 |
| 0-5 mm | 508 | Plain 3D U-Net 300 ep | 0.063 [0.04, 0.09] | 0.421 | 0.051 | 0.001 | 44 |
| 0-5 mm | 508 | U-Net + synthetic small tumours | 0.122 [0.10, 0.15] | 0.218 | 0.076 | 0.011 | 223 |
| 0-5 mm | 508 | U-Net + CC-DiceCE (w = 0.5) | 0.081 [0.06, 0.11] | 0.461 | 0.062 | 0.001 | 48 |
| 5-10 mm | 458 | nnU-Net 300 ep | 0.079 [0.06, 0.11] | 0.375 | 0.068 | 0.028 | 60 |
| 5-10 mm | 458 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.346 | 0.059 | 0.020 | 53 |
| 5-10 mm | 458 | U-Net + synthetic small tumours | 0.229 [0.19, 0.27] | 0.316 | 0.161 | 0.111 | 227 |
| 5-10 mm | 458 | U-Net + CC-DiceCE (w = 0.5) | 0.100 [0.08, 0.13] | 0.407 | 0.095 | 0.039 | 67 |
| 10-15 mm | 323 | nnU-Net 300 ep | 0.440 [0.39, 0.49] | 0.798 | 0.318 | 0.244 | 36 |
| 10-15 mm | 323 | Plain 3D U-Net 300 ep | 0.455 [0.40, 0.51] | 0.817 | 0.330 | 0.249 | 33 |
| 10-15 mm | 323 | U-Net + synthetic small tumours | 0.622 [0.57, 0.67] | 0.728 | 0.421 | 0.368 | 75 |
| 10-15 mm | 323 | U-Net + CC-DiceCE (w = 0.5) | 0.542 [0.49, 0.60] | 0.754 | 0.410 | 0.298 | 57 |
| > 15 mm | 645 | nnU-Net 300 ep | 0.878 [0.85, 0.90] | 0.932 | 0.802 | 0.651 | 41 |
| > 15 mm | 645 | Plain 3D U-Net 300 ep | 0.865 [0.84, 0.89] | 0.938 | 0.830 | 0.643 | 37 |
| > 15 mm | 645 | U-Net + synthetic small tumours | 0.893 [0.87, 0.91] | 0.943 | 0.833 | 0.666 | 35 |
| > 15 mm | 645 | U-Net + CC-DiceCE (w = 0.5) | 0.902 [0.88, 0.92] | 0.937 | 0.834 | 0.665 | 39 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 1 | nnU-Net 300 ep | 1.000 [0.21, 1.00] | 0.091 | 1.000 | 0.000 | 10 |
| 0-5 mm | 1 | Plain 3D U-Net 300 ep | 1.000 [0.21, 1.00] | 0.067 | 1.000 | 0.000 | 14 |
| 0-5 mm | 1 | U-Net + synthetic small tumours | 1.000 [0.21, 1.00] | 0.011 | 1.000 | 0.000 | 93 |
| 0-5 mm | 1 | U-Net + CC-DiceCE (w = 0.5) | 1.000 [0.21, 1.00] | 0.077 | 1.000 | 0.000 | 12 |
| 5-10 mm | 9 | nnU-Net 300 ep | 0.111 [0.02, 0.44] | 0.333 | 0.061 | 0.000 | 2 |
| 5-10 mm | 9 | Plain 3D U-Net 300 ep | 0.111 [0.02, 0.44] | 0.143 | 0.087 | 0.000 | 6 |
| 5-10 mm | 9 | U-Net + synthetic small tumours | 0.889 [0.56, 0.98] | 0.174 | 0.508 | 0.382 | 38 |
| 5-10 mm | 9 | U-Net + CC-DiceCE (w = 0.5) | 0.444 [0.19, 0.73] | 0.250 | 0.222 | 0.113 | 12 |
| 10-15 mm | 23 | nnU-Net 300 ep | 0.391 [0.22, 0.59] | 0.750 | 0.333 | 0.229 | 3 |
| 10-15 mm | 23 | Plain 3D U-Net 300 ep | 0.435 [0.26, 0.63] | 0.588 | 0.306 | 0.232 | 7 |
| 10-15 mm | 23 | U-Net + synthetic small tumours | 0.696 [0.49, 0.84] | 0.533 | 0.522 | 0.420 | 14 |
| 10-15 mm | 23 | U-Net + CC-DiceCE (w = 0.5) | 0.609 [0.41, 0.78] | 0.609 | 0.486 | 0.330 | 9 |
| > 15 mm | 49 | nnU-Net 300 ep | 0.837 [0.71, 0.91] | 0.854 | 0.764 | 0.564 | 7 |
| > 15 mm | 49 | Plain 3D U-Net 300 ep | 0.857 [0.73, 0.93] | 0.913 | 0.836 | 0.566 | 4 |
| > 15 mm | 49 | U-Net + synthetic small tumours | 0.837 [0.71, 0.91] | 0.911 | 0.845 | 0.561 | 4 |
| > 15 mm | 49 | U-Net + CC-DiceCE (w = 0.5) | 0.857 [0.73, 0.93] | 0.894 | 0.851 | 0.561 | 5 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 22 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 13 |
| 5-10 mm | 8 | nnU-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 8 | Plain 3D U-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 8 | U-Net + synthetic small tumours | 0.250 [0.07, 0.59] | 0.095 | 0.114 | 0.105 | 19 |
| 5-10 mm | 8 | U-Net + CC-DiceCE (w = 0.5) | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 19 | nnU-Net 300 ep | 0.211 [0.09, 0.43] | 0.667 | 0.227 | 0.141 | 2 |
| 10-15 mm | 19 | Plain 3D U-Net 300 ep | 0.211 [0.09, 0.43] | 0.500 | 0.219 | 0.134 | 4 |
| 10-15 mm | 19 | U-Net + synthetic small tumours | 0.421 [0.23, 0.64] | 0.571 | 0.206 | 0.215 | 6 |
| 10-15 mm | 19 | U-Net + CC-DiceCE (w = 0.5) | 0.211 [0.09, 0.43] | 0.667 | 0.218 | 0.131 | 2 |
| > 15 mm | 48 | nnU-Net 300 ep | 0.812 [0.68, 0.90] | 0.886 | 0.810 | 0.655 | 5 |
| > 15 mm | 48 | Plain 3D U-Net 300 ep | 0.833 [0.70, 0.91] | 0.870 | 0.813 | 0.675 | 6 |
| > 15 mm | 48 | U-Net + synthetic small tumours | 0.875 [0.75, 0.94] | 0.913 | 0.831 | 0.690 | 4 |
| > 15 mm | 48 | U-Net + CC-DiceCE (w = 0.5) | 0.875 [0.75, 0.94] | 0.824 | 0.829 | 0.688 | 9 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + synthetic small tumours: {"epochs_logged": 300, "mean_epoch_seconds": 48.1, "final_val_patch_dice_liver_tumour": [0.9574, 0.8441], "best_val_patch_dice_tumour": 0.8728, "final_train_loss": 0.2705, "synthetic_lesions_per_epoch_mean": 496.0}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | U-Net + synthetic small tumours | U-Net + CC-DiceCE (w = 0.5) |
|---|---:|---|---:|---:|---:|---:|
| overall | 67 | case tumour Dice (mean) | 0.713 | 0.702 | 0.652 | 0.670 |
|  |  | voxel Dice (pooled) | 0.807 | 0.784 | 0.786 | 0.782 |
|  |  | voxel recall (pooled) | 0.757 | 0.753 | 0.772 | 0.764 |
|  |  | voxel precision (pooled) | 0.865 | 0.818 | 0.801 | 0.802 |
|  |  | lesion recall = sensitivity | 0.973 | 0.959 | 0.959 | 0.932 |
|  |  | lesion precision | 0.626 | 0.617 | 0.314 | 0.473 |
|  |  | lesion F1 | 0.762 | 0.751 | 0.473 | 0.627 |
|  |  | false positives per case | 0.64 | 0.66 | 2.31 | 1.15 |
| MCT-LTDiag | 19 | case tumour Dice (mean) | 0.788 | 0.788 | 0.768 | 0.787 |
|  |  | voxel Dice (pooled) | 0.863 | 0.859 | 0.835 | 0.866 |
|  |  | voxel recall (pooled) | 0.854 | 0.851 | 0.818 | 0.883 |
|  |  | voxel precision (pooled) | 0.872 | 0.868 | 0.854 | 0.850 |
|  |  | lesion recall = sensitivity | 0.960 | 0.960 | 0.960 | 0.960 |
|  |  | lesion precision | 0.750 | 0.774 | 0.393 | 0.706 |
|  |  | lesion F1 | 0.842 | 0.857 | 0.558 | 0.814 |
|  |  | false positives per case | 0.42 | 0.37 | 1.95 | 0.53 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.642 | 0.614 | 0.534 | 0.571 |
|  |  | voxel Dice (pooled) | 0.797 | 0.751 | 0.760 | 0.744 |
|  |  | voxel recall (pooled) | 0.742 | 0.734 | 0.769 | 0.738 |
|  |  | voxel precision (pooled) | 0.861 | 0.770 | 0.751 | 0.751 |
|  |  | lesion recall = sensitivity | 0.963 | 0.926 | 0.926 | 0.889 |
|  |  | lesion precision | 0.553 | 0.446 | 0.202 | 0.282 |
|  |  | lesion F1 | 0.703 | 0.602 | 0.331 | 0.429 |
|  |  | false positives per case | 0.62 | 0.91 | 2.91 | 1.79 |
| WAW-TACE | 14 | case tumour Dice (mean) | 0.747 | 0.757 | 0.749 | 0.707 |
|  |  | voxel Dice (pooled) | 0.788 | 0.817 | 0.824 | 0.822 |
|  |  | voxel recall (pooled) | 0.721 | 0.729 | 0.745 | 0.742 |
|  |  | voxel precision (pooled) | 0.868 | 0.929 | 0.922 | 0.922 |
|  |  | lesion recall = sensitivity | 1.000 | 1.000 | 1.000 | 0.955 |
|  |  | lesion precision | 0.611 | 0.786 | 0.537 | 0.778 |
|  |  | lesion F1 | 0.759 | 0.880 | 0.698 | 0.857 |
|  |  | false positives per case | 1.00 | 0.43 | 1.36 | 0.43 |

### Per size bin

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 18 |
| 0-5 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 71 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 26 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 10 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 54 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 18 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 22 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 21 |
| > 15 mm | 74 | nnU-Net 300 ep | 0.973 [0.91, 0.99] | 0.889 | 0.757 | 0.727 | 9 |
| > 15 mm | 74 | Plain 3D U-Net 300 ep | 0.959 [0.89, 0.99] | 0.910 | 0.753 | 0.726 | 7 |
| > 15 mm | 74 | U-Net + synthetic small tumours | 0.959 [0.89, 0.99] | 0.899 | 0.772 | 0.705 | 8 |
| > 15 mm | 74 | U-Net + CC-DiceCE (w = 0.5) | 0.932 [0.85, 0.97] | 0.852 | 0.764 | 0.708 | 12 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 14 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 8 |
| > 15 mm | 25 | nnU-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.854 | 0.772 | 0 |
| > 15 mm | 25 | Plain 3D U-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.851 | 0.776 | 0 |
| > 15 mm | 25 | U-Net + synthetic small tumours | 0.960 [0.80, 0.99] | 0.960 | 0.818 | 0.768 | 1 |
| > 15 mm | 25 | U-Net + CC-DiceCE (w = 0.5) | 0.960 [0.80, 0.99] | 0.923 | 0.883 | 0.783 | 2 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 0-5 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 48 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 24 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 33 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 17 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 11 |
| > 15 mm | 27 | nnU-Net 300 ep | 0.963 [0.82, 0.99] | 0.788 | 0.742 | 0.694 | 7 |
| > 15 mm | 27 | Plain 3D U-Net 300 ep | 0.926 [0.77, 0.98] | 0.781 | 0.734 | 0.681 | 7 |
| > 15 mm | 27 | U-Net + synthetic small tumours | 0.926 [0.77, 0.98] | 0.806 | 0.769 | 0.661 | 6 |
| > 15 mm | 27 | U-Net + CC-DiceCE (w = 0.5) | 0.889 [0.72, 0.96] | 0.727 | 0.738 | 0.648 | 9 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 10 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 7 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| > 15 mm | 22 | nnU-Net 300 ep | 1.000 [0.85, 1.00] | 0.917 | 0.721 | 0.716 | 2 |
| > 15 mm | 22 | Plain 3D U-Net 300 ep | 1.000 [0.85, 1.00] | 1.000 | 0.729 | 0.724 | 0 |
| > 15 mm | 22 | U-Net + synthetic small tumours | 1.000 [0.85, 1.00] | 0.957 | 0.745 | 0.686 | 1 |
| > 15 mm | 22 | U-Net + CC-DiceCE (w = 0.5) | 0.955 [0.78, 0.99] | 0.955 | 0.742 | 0.697 | 1 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + synthetic small tumours: {"epochs_logged": 300, "mean_epoch_seconds": 48.1, "final_val_patch_dice_liver_tumour": [0.9574, 0.8441], "best_val_patch_dice_tumour": 0.8728, "final_train_loss": 0.2705, "synthetic_lesions_per_epoch_mean": 496.0}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Artefacts

`results/2026-09-28_unet_cc_dicece_loss/`: ccdicece_config.json, ccdicece_history.json, ccdicece_jobs.json, ccdicece_test_per_lesion.csv, ccdicece_test_size_binned_metrics.csv, ccdicece_val_per_lesion.csv, ccdicece_val_size_binned_metrics.csv, nnunet_cohorts.json, nnunet_nnUNetPlans.json, nnunet_test_per_lesion.csv, nnunet_test_size_binned_metrics.csv, nnunet_training_log_2026_9_27_02_50_22.txt, nnunet_val_per_lesion.csv, nnunet_val_size_binned_metrics.csv, synth_config.json, synth_history.json, synth_jobs.json, synth_test_per_lesion.csv, synth_test_size_binned_metrics.csv, synth_val_per_lesion.csv, synth_val_size_binned_metrics.csv, unet3d_config.json, unet3d_history.json, unet3d_test_per_lesion.csv, unet3d_test_size_binned_metrics.csv, unet3d_val_per_lesion.csv, unet3d_val_size_binned_metrics.csv

