# 2026-09-27: baselines small held out

**Purpose.** Establish the two reference baselines on unified_v2: the standard nnU-Net v2 (official trainer, 300 epochs) and a plain 3D U-Net built from the same nnU-Net plans (identical architecture and parameter count), both trained only on patients whose lesions are all > 15 mm and evaluated on the held-out small-lesion patients, to measure how far learning from large lesions transfers to small ones.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Labels: 0 background, 1 liver, 2 tumour. Split: small-held-out seed 0 (train 534 / val 67 / test 439); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM (`job_chain.json`).

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep |
|---|---:|---|---:|---:|
| overall | 439 | case tumour Dice (mean) | 0.713 | 0.709 |
|  |  | voxel Dice (pooled) | 0.828 | 0.851 |
|  |  | voxel recall (pooled) | 0.795 | 0.827 |
|  |  | voxel precision (pooled) | 0.865 | 0.877 |
|  |  | lesion recall = sensitivity | 0.413 | 0.413 |
|  |  | lesion precision | 0.786 | 0.797 |
|  |  | lesion F1 | 0.542 | 0.544 |
|  |  | false positives per case | 0.54 | 0.50 |
| MCT-LTDiag | 354 | case tumour Dice (mean) | 0.733 | 0.730 |
|  |  | voxel Dice (pooled) | 0.835 | 0.856 |
|  |  | voxel recall (pooled) | 0.799 | 0.827 |
|  |  | voxel precision (pooled) | 0.874 | 0.887 |
|  |  | lesion recall = sensitivity | 0.398 | 0.396 |
|  |  | lesion precision | 0.799 | 0.821 |
|  |  | lesion F1 | 0.531 | 0.534 |
|  |  | false positives per case | 0.55 | 0.47 |
| PLC-CECT | 48 | case tumour Dice (mean) | 0.540 | 0.537 |
|  |  | voxel Dice (pooled) | 0.801 | 0.843 |
|  |  | voxel recall (pooled) | 0.763 | 0.835 |
|  |  | voxel precision (pooled) | 0.843 | 0.852 |
|  |  | lesion recall = sensitivity | 0.634 | 0.659 |
|  |  | lesion precision | 0.703 | 0.635 |
|  |  | lesion F1 | 0.667 | 0.647 |
|  |  | false positives per case | 0.46 | 0.65 |
| WAW-TACE | 37 | case tumour Dice (mean) | 0.727 | 0.717 |
|  |  | voxel Dice (pooled) | 0.801 | 0.800 |
|  |  | voxel recall (pooled) | 0.807 | 0.810 |
|  |  | voxel precision (pooled) | 0.795 | 0.790 |
|  |  | lesion recall = sensitivity | 0.573 | 0.587 |
|  |  | lesion precision | 0.683 | 0.667 |
|  |  | lesion F1 | 0.623 | 0.624 |
|  |  | false positives per case | 0.54 | 0.59 |

### Per size bin

#### overall

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 509 | nnU-Net 300 ep | 0.051 [0.04, 0.07] | 0.252 | 0.048 | 0.000 | 77 |
| 0-5 mm | 509 | Plain 3D U-Net 300 ep | 0.065 [0.05, 0.09] | 0.337 | 0.051 | 0.001 | 65 |
| 5-10 mm | 475 | nnU-Net 300 ep | 0.078 [0.06, 0.11] | 0.366 | 0.066 | 0.027 | 64 |
| 5-10 mm | 475 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.312 | 0.058 | 0.019 | 64 |
| 10-15 mm | 365 | nnU-Net 300 ep | 0.425 [0.37, 0.48] | 0.791 | 0.314 | 0.237 | 41 |
| 10-15 mm | 365 | Plain 3D U-Net 300 ep | 0.441 [0.39, 0.49] | 0.785 | 0.322 | 0.242 | 44 |
| > 15 mm | 742 | nnU-Net 300 ep | 0.871 [0.84, 0.89] | 0.924 | 0.797 | 0.646 | 53 |
| > 15 mm | 742 | Plain 3D U-Net 300 ep | 0.863 [0.84, 0.89] | 0.932 | 0.830 | 0.640 | 47 |

#### MCT-LTDiag

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 508 | nnU-Net 300 ep | 0.049 [0.03, 0.07] | 0.309 | 0.048 | 0.000 | 56 |
| 0-5 mm | 508 | Plain 3D U-Net 300 ep | 0.063 [0.04, 0.09] | 0.421 | 0.051 | 0.001 | 44 |
| 5-10 mm | 458 | nnU-Net 300 ep | 0.079 [0.06, 0.11] | 0.375 | 0.068 | 0.028 | 60 |
| 5-10 mm | 458 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.346 | 0.059 | 0.020 | 53 |
| 10-15 mm | 323 | nnU-Net 300 ep | 0.440 [0.39, 0.49] | 0.798 | 0.318 | 0.244 | 36 |
| 10-15 mm | 323 | Plain 3D U-Net 300 ep | 0.455 [0.40, 0.51] | 0.817 | 0.330 | 0.249 | 33 |
| > 15 mm | 645 | nnU-Net 300 ep | 0.878 [0.85, 0.90] | 0.932 | 0.802 | 0.651 | 41 |
| > 15 mm | 645 | Plain 3D U-Net 300 ep | 0.865 [0.84, 0.89] | 0.938 | 0.830 | 0.643 | 37 |

#### PLC-CECT

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 1 | nnU-Net 300 ep | 1.000 [0.21, 1.00] | 0.091 | 1.000 | 0.000 | 10 |
| 0-5 mm | 1 | Plain 3D U-Net 300 ep | 1.000 [0.21, 1.00] | 0.067 | 1.000 | 0.000 | 14 |
| 5-10 mm | 9 | nnU-Net 300 ep | 0.111 [0.02, 0.44] | 0.333 | 0.061 | 0.000 | 2 |
| 5-10 mm | 9 | Plain 3D U-Net 300 ep | 0.111 [0.02, 0.44] | 0.143 | 0.087 | 0.000 | 6 |
| 10-15 mm | 23 | nnU-Net 300 ep | 0.391 [0.22, 0.59] | 0.750 | 0.333 | 0.229 | 3 |
| 10-15 mm | 23 | Plain 3D U-Net 300 ep | 0.435 [0.26, 0.63] | 0.588 | 0.306 | 0.232 | 7 |
| > 15 mm | 49 | nnU-Net 300 ep | 0.837 [0.71, 0.91] | 0.854 | 0.764 | 0.564 | 7 |
| > 15 mm | 49 | Plain 3D U-Net 300 ep | 0.857 [0.73, 0.93] | 0.913 | 0.836 | 0.566 | 4 |

#### WAW-TACE

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 7 |
| 5-10 mm | 8 | nnU-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 8 | Plain 3D U-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 19 | nnU-Net 300 ep | 0.211 [0.09, 0.43] | 0.667 | 0.227 | 0.141 | 2 |
| 10-15 mm | 19 | Plain 3D U-Net 300 ep | 0.211 [0.09, 0.43] | 0.500 | 0.219 | 0.134 | 4 |
| > 15 mm | 48 | nnU-Net 300 ep | 0.812 [0.68, 0.90] | 0.886 | 0.810 | 0.655 | 5 |
| > 15 mm | 48 | Plain 3D U-Net 300 ep | 0.833 [0.70, 0.91] | 0.870 | 0.813 | 0.675 | 6 |

### Training

```
nnU-Net:        {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
shared architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep |
|---|---:|---|---:|---:|
| overall | 67 | case tumour Dice (mean) | 0.713 | 0.702 |
|  |  | voxel Dice (pooled) | 0.807 | 0.784 |
|  |  | voxel recall (pooled) | 0.757 | 0.753 |
|  |  | voxel precision (pooled) | 0.865 | 0.818 |
|  |  | lesion recall = sensitivity | 0.973 | 0.959 |
|  |  | lesion precision | 0.626 | 0.617 |
|  |  | lesion F1 | 0.762 | 0.751 |
|  |  | false positives per case | 0.64 | 0.66 |
| MCT-LTDiag | 19 | case tumour Dice (mean) | 0.788 | 0.788 |
|  |  | voxel Dice (pooled) | 0.863 | 0.859 |
|  |  | voxel recall (pooled) | 0.854 | 0.851 |
|  |  | voxel precision (pooled) | 0.872 | 0.868 |
|  |  | lesion recall = sensitivity | 0.960 | 0.960 |
|  |  | lesion precision | 0.750 | 0.774 |
|  |  | lesion F1 | 0.842 | 0.857 |
|  |  | false positives per case | 0.42 | 0.37 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.642 | 0.614 |
|  |  | voxel Dice (pooled) | 0.797 | 0.751 |
|  |  | voxel recall (pooled) | 0.742 | 0.734 |
|  |  | voxel precision (pooled) | 0.861 | 0.770 |
|  |  | lesion recall = sensitivity | 0.963 | 0.926 |
|  |  | lesion precision | 0.553 | 0.446 |
|  |  | lesion F1 | 0.703 | 0.602 |
|  |  | false positives per case | 0.62 | 0.91 |
| WAW-TACE | 14 | case tumour Dice (mean) | 0.747 | 0.757 |
|  |  | voxel Dice (pooled) | 0.788 | 0.817 |
|  |  | voxel recall (pooled) | 0.721 | 0.729 |
|  |  | voxel precision (pooled) | 0.868 | 0.929 |
|  |  | lesion recall = sensitivity | 1.000 | 1.000 |
|  |  | lesion precision | 0.611 | 0.786 |
|  |  | lesion F1 | 0.759 | 0.880 |
|  |  | false positives per case | 1.00 | 0.43 |

### Per size bin

#### overall

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 18 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 10 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| > 15 mm | 74 | nnU-Net 300 ep | 0.973 [0.91, 0.99] | 0.889 | 0.757 | 0.727 | 9 |
| > 15 mm | 74 | Plain 3D U-Net 300 ep | 0.959 [0.89, 0.99] | 0.910 | 0.753 | 0.726 | 7 |

#### MCT-LTDiag

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| > 15 mm | 25 | nnU-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.854 | 0.772 | 0 |
| > 15 mm | 25 | Plain 3D U-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.851 | 0.776 | 0 |

#### PLC-CECT

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| > 15 mm | 27 | nnU-Net 300 ep | 0.963 [0.82, 0.99] | 0.788 | 0.742 | 0.694 | 7 |
| > 15 mm | 27 | Plain 3D U-Net 300 ep | 0.926 [0.77, 0.98] | 0.781 | 0.734 | 0.681 | 7 |

#### WAW-TACE

| Size bin | Lesions | Baseline | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| > 15 mm | 22 | nnU-Net 300 ep | 1.000 [0.85, 1.00] | 0.917 | 0.721 | 0.716 | 2 |
| > 15 mm | 22 | Plain 3D U-Net 300 ep | 1.000 [0.85, 1.00] | 1.000 | 0.729 | 0.724 | 0 |

### Training

```
nnU-Net:        {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
shared architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Artefacts

`results/2026-09-27_baselines_small_held_out/`: job_chain.json, nnUNetPlans.json, nnunet_cohorts.json, nnunet_test_per_lesion.csv, nnunet_test_size_binned_metrics.csv, nnunet_training_log_2026_9_27_02_50_22.txt, nnunet_val_per_lesion.csv, nnunet_val_size_binned_metrics.csv, unet3d_config.json, unet3d_history.json, unet3d_test_per_lesion.csv, unet3d_test_size_binned_metrics.csv, unet3d_val_per_lesion.csv, unet3d_val_size_binned_metrics.csv

