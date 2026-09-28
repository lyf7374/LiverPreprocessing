# 2026-09-28: plan stage1 segmentation branch

**Purpose.** Reproduction of PLAN (Yan et al., MICCAI 2023, github.com/alibaba-damo-academy/pixel-lesion-patient-network) on unified_v2, stage 1 of 2: the pixel branch alone (nnU-Net v1 encoder + FPN pixel decoder + segmentation head, PLAN's data loader, optimiser and post-processing). Stage 2 (pixel + lesion + patient branches, initialised from this run) is training. Compared with our two 300-epoch baselines.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: small-held-out seed 0 (train 534 / val 67 / test 439); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

PLAN's released code (nnU-Net v1 fork) with our data exposed as nnU-Net v1 task Task520_HCC (labels 1 tumour, 2 liver as PLAN expects; images identical to the baselines, including the PLC intensity offset; same train/val/test cases). Changes to the released code needed to run it: torch.load(weights_only=False) for torch 2.8, and the trainer's missing imports of DataLoader3D_Inst, PLAN and Mask2Former (resolved by name). Config cfg_hcc_plan1 mirrors cfg_dce_plan1: one lesion class, liver as organ class 2, components ('seg',), FPN pixel decoder with 64-channel mask embeddings, RAdam 1e-4, PLAN's instance data loader (foreground sampling on lesions, lesions < 10 voxels ignored), 300 epochs x 250 iterations instead of 500 (budget matched to the baselines), cudnn deterministic mode off. nnU-Net v1 planning gave the same configuration as v2: patch 112 x 128 x 160, batch 2, 1 mm, 32 base features, 6 stages. Inference with PLAN's pipeline: sliding window, mirroring, then PLAN's post-processing (largest liver component, lesions farther than 3 mm from the liver removed, minimum lesion volume 25 mm3); the raw and the post-processed label maps were both evaluated with our evaluator (tumour label 1). Training 21.7 h (4.3 min per epoch), inference + evaluation 1.3 h (job 29076121).

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `nnunet` = nnU-Net 300 ep; `unet3d` = Plain 3D U-Net 300 ep; `plan1` = PLAN stage 1 (seg branch) 300 ep

**Findings.**

PLAN's pixel branch is on par with, or slightly better than, the baselines: case tumour Dice 0.716 (nnU-Net 0.713, plain U-Net 0.709), overall lesion recall 0.455 vs 0.413 at the same lesion precision 0.797, F1 0.579 (the best single number so far), false positives per case 0.55; by size 0-5 mm 0.073, 5-10 mm 0.116, 10-15 mm 0.512, > 15 mm 0.906 (plain U-Net 0.065 / 0.061 / 0.441 / 0.863). Per dataset: PLC-CECT case Dice 0.567 (0.537) and recall 0.683 (0.659); WAW-TACE recall 0.667 (0.587) at 0.49 FP per case. Pooled voxel Dice is lower than the plain U-Net (0.834 vs 0.851) because voxel recall is lower (0.801 vs 0.827): PLAN's lesion-oriented sampling and post-processing favour detecting components over covering them. PLAN's post-processing matters for precision only: without it the same predictions give recall 0.456, precision 0.669, 1.07 false positives per case (0-5 mm bin: 225 vs 42 false-positive components), with it precision 0.797 and 0.55 per case; sensitivity per bin is unchanged, so the removed components are almost all false positives. Note the different optimiser and sampling (RAdam 1e-4 with lesion-forced foreground sampling vs SGD 0.01 with 33 % foreground); the difference to the baselines is of the same order as the CC-DiceCE gain in experiment 2.

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | PLAN stage 1 (seg branch) 300 ep |
|---|---:|---|---:|---:|---:|
| overall | 439 | case tumour Dice (mean) | 0.713 | 0.709 | 0.716 |
|  |  | voxel Dice (pooled) | 0.828 | 0.851 | 0.834 |
|  |  | voxel recall (pooled) | 0.795 | 0.827 | 0.801 |
|  |  | voxel precision (pooled) | 0.865 | 0.877 | 0.870 |
|  |  | lesion recall = sensitivity | 0.413 | 0.413 | 0.455 |
|  |  | lesion precision | 0.786 | 0.797 | 0.797 |
|  |  | lesion F1 | 0.542 | 0.544 | 0.579 |
|  |  | false positives per case | 0.54 | 0.50 | 0.55 |
| MCT-LTDiag | 354 | case tumour Dice (mean) | 0.733 | 0.730 | 0.734 |
|  |  | voxel Dice (pooled) | 0.835 | 0.856 | 0.838 |
|  |  | voxel recall (pooled) | 0.799 | 0.827 | 0.798 |
|  |  | voxel precision (pooled) | 0.874 | 0.887 | 0.882 |
|  |  | lesion recall = sensitivity | 0.398 | 0.396 | 0.437 |
|  |  | lesion precision | 0.799 | 0.821 | 0.813 |
|  |  | lesion F1 | 0.531 | 0.534 | 0.568 |
|  |  | false positives per case | 0.55 | 0.47 | 0.55 |
| PLC-CECT | 48 | case tumour Dice (mean) | 0.540 | 0.537 | 0.567 |
|  |  | voxel Dice (pooled) | 0.801 | 0.843 | 0.832 |
|  |  | voxel recall (pooled) | 0.763 | 0.835 | 0.802 |
|  |  | voxel precision (pooled) | 0.843 | 0.852 | 0.866 |
|  |  | lesion recall = sensitivity | 0.634 | 0.659 | 0.683 |
|  |  | lesion precision | 0.703 | 0.635 | 0.651 |
|  |  | lesion F1 | 0.667 | 0.647 | 0.667 |
|  |  | false positives per case | 0.46 | 0.65 | 0.62 |
| WAW-TACE | 37 | case tumour Dice (mean) | 0.727 | 0.717 | 0.718 |
|  |  | voxel Dice (pooled) | 0.801 | 0.800 | 0.783 |
|  |  | voxel recall (pooled) | 0.807 | 0.810 | 0.837 |
|  |  | voxel precision (pooled) | 0.795 | 0.790 | 0.736 |
|  |  | lesion recall = sensitivity | 0.573 | 0.587 | 0.667 |
|  |  | lesion precision | 0.683 | 0.667 | 0.735 |
|  |  | lesion F1 | 0.623 | 0.624 | 0.699 |
|  |  | false positives per case | 0.54 | 0.59 | 0.49 |

### Per size bin

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 509 | nnU-Net 300 ep | 0.051 [0.04, 0.07] | 0.252 | 0.048 | 0.000 | 77 |
| 0-5 mm | 509 | Plain 3D U-Net 300 ep | 0.065 [0.05, 0.09] | 0.337 | 0.051 | 0.001 | 65 |
| 0-5 mm | 509 | PLAN stage 1 (seg branch) 300 ep | 0.073 [0.05, 0.10] | 0.468 | 0.060 | 0.001 | 42 |
| 5-10 mm | 475 | nnU-Net 300 ep | 0.078 [0.06, 0.11] | 0.366 | 0.066 | 0.027 | 64 |
| 5-10 mm | 475 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.312 | 0.058 | 0.019 | 64 |
| 5-10 mm | 475 | PLAN stage 1 (seg branch) 300 ep | 0.116 [0.09, 0.15] | 0.348 | 0.109 | 0.045 | 103 |
| 10-15 mm | 365 | nnU-Net 300 ep | 0.425 [0.37, 0.48] | 0.791 | 0.314 | 0.237 | 41 |
| 10-15 mm | 365 | Plain 3D U-Net 300 ep | 0.441 [0.39, 0.49] | 0.785 | 0.322 | 0.242 | 44 |
| 10-15 mm | 365 | PLAN stage 1 (seg branch) 300 ep | 0.512 [0.46, 0.56] | 0.760 | 0.371 | 0.280 | 59 |
| > 15 mm | 742 | nnU-Net 300 ep | 0.871 [0.84, 0.89] | 0.924 | 0.797 | 0.646 | 53 |
| > 15 mm | 742 | Plain 3D U-Net 300 ep | 0.863 [0.84, 0.89] | 0.932 | 0.830 | 0.640 | 47 |
| > 15 mm | 742 | PLAN stage 1 (seg branch) 300 ep | 0.906 [0.88, 0.92] | 0.946 | 0.803 | 0.668 | 38 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 508 | nnU-Net 300 ep | 0.049 [0.03, 0.07] | 0.309 | 0.048 | 0.000 | 56 |
| 0-5 mm | 508 | Plain 3D U-Net 300 ep | 0.063 [0.04, 0.09] | 0.421 | 0.051 | 0.001 | 44 |
| 0-5 mm | 508 | PLAN stage 1 (seg branch) 300 ep | 0.071 [0.05, 0.10] | 0.507 | 0.060 | 0.001 | 35 |
| 5-10 mm | 458 | nnU-Net 300 ep | 0.079 [0.06, 0.11] | 0.375 | 0.068 | 0.028 | 60 |
| 5-10 mm | 458 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.346 | 0.059 | 0.020 | 53 |
| 5-10 mm | 458 | PLAN stage 1 (seg branch) 300 ep | 0.116 [0.09, 0.15] | 0.384 | 0.110 | 0.045 | 85 |
| 10-15 mm | 323 | nnU-Net 300 ep | 0.440 [0.39, 0.49] | 0.798 | 0.318 | 0.244 | 36 |
| 10-15 mm | 323 | Plain 3D U-Net 300 ep | 0.455 [0.40, 0.51] | 0.817 | 0.330 | 0.249 | 33 |
| 10-15 mm | 323 | PLAN stage 1 (seg branch) 300 ep | 0.526 [0.47, 0.58] | 0.776 | 0.379 | 0.288 | 49 |
| > 15 mm | 645 | nnU-Net 300 ep | 0.878 [0.85, 0.90] | 0.932 | 0.802 | 0.651 | 41 |
| > 15 mm | 645 | Plain 3D U-Net 300 ep | 0.865 [0.84, 0.89] | 0.938 | 0.830 | 0.643 | 37 |
| > 15 mm | 645 | PLAN stage 1 (seg branch) 300 ep | 0.909 [0.88, 0.93] | 0.959 | 0.801 | 0.672 | 25 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 1 | nnU-Net 300 ep | 1.000 [0.21, 1.00] | 0.091 | 1.000 | 0.000 | 10 |
| 0-5 mm | 1 | Plain 3D U-Net 300 ep | 1.000 [0.21, 1.00] | 0.067 | 1.000 | 0.000 | 14 |
| 0-5 mm | 1 | PLAN stage 1 (seg branch) 300 ep | 1.000 [0.21, 1.00] | 0.250 | 1.000 | 0.000 | 3 |
| 5-10 mm | 9 | nnU-Net 300 ep | 0.111 [0.02, 0.44] | 0.333 | 0.061 | 0.000 | 2 |
| 5-10 mm | 9 | Plain 3D U-Net 300 ep | 0.111 [0.02, 0.44] | 0.143 | 0.087 | 0.000 | 6 |
| 5-10 mm | 9 | PLAN stage 1 (seg branch) 300 ep | 0.111 [0.02, 0.44] | 0.067 | 0.117 | 0.000 | 14 |
| 10-15 mm | 23 | nnU-Net 300 ep | 0.391 [0.22, 0.59] | 0.750 | 0.333 | 0.229 | 3 |
| 10-15 mm | 23 | Plain 3D U-Net 300 ep | 0.435 [0.26, 0.63] | 0.588 | 0.306 | 0.232 | 7 |
| 10-15 mm | 23 | PLAN stage 1 (seg branch) 300 ep | 0.478 [0.29, 0.67] | 0.611 | 0.384 | 0.261 | 7 |
| > 15 mm | 49 | nnU-Net 300 ep | 0.837 [0.71, 0.91] | 0.854 | 0.764 | 0.564 | 7 |
| > 15 mm | 49 | Plain 3D U-Net 300 ep | 0.857 [0.73, 0.93] | 0.913 | 0.836 | 0.566 | 4 |
| > 15 mm | 49 | PLAN stage 1 (seg branch) 300 ep | 0.878 [0.76, 0.94] | 0.878 | 0.803 | 0.600 | 6 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 8 | nnU-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 8 | Plain 3D U-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 8 | PLAN stage 1 (seg branch) 300 ep | 0.125 [0.02, 0.47] | 0.200 | 0.082 | 0.073 | 4 |
| 10-15 mm | 19 | nnU-Net 300 ep | 0.211 [0.09, 0.43] | 0.667 | 0.227 | 0.141 | 2 |
| 10-15 mm | 19 | Plain 3D U-Net 300 ep | 0.211 [0.09, 0.43] | 0.500 | 0.219 | 0.134 | 4 |
| 10-15 mm | 19 | PLAN stage 1 (seg branch) 300 ep | 0.316 [0.15, 0.54] | 0.667 | 0.230 | 0.167 | 3 |
| > 15 mm | 48 | nnU-Net 300 ep | 0.812 [0.68, 0.90] | 0.886 | 0.810 | 0.655 | 5 |
| > 15 mm | 48 | Plain 3D U-Net 300 ep | 0.833 [0.70, 0.91] | 0.870 | 0.813 | 0.675 | 6 |
| > 15 mm | 48 | PLAN stage 1 (seg branch) 300 ep | 0.896 [0.78, 0.95] | 0.860 | 0.840 | 0.688 | 7 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
PLAN stage 1 (seg branch) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.4165, "final_val_loss": -1.2835, "final_global_dice_tumour_liver": [0.8185, 0.9638], "final_online_lesion_sens_prec": [0.871, 0.4426], "training_hours": 21.7}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [183, 218, 246], "current_spacing": [1.0, 1.0, 1.0], "do_dummy_2D_data_aug": false, "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | PLAN stage 1 (seg branch) 300 ep |
|---|---:|---|---:|---:|---:|
| overall | 67 | case tumour Dice (mean) | 0.713 | 0.702 | 0.696 |
|  |  | voxel Dice (pooled) | 0.807 | 0.784 | 0.806 |
|  |  | voxel recall (pooled) | 0.757 | 0.753 | 0.773 |
|  |  | voxel precision (pooled) | 0.865 | 0.818 | 0.842 |
|  |  | lesion recall = sensitivity | 0.973 | 0.959 | 0.959 |
|  |  | lesion precision | 0.626 | 0.617 | 0.607 |
|  |  | lesion F1 | 0.762 | 0.751 | 0.743 |
|  |  | false positives per case | 0.64 | 0.66 | 0.69 |
| MCT-LTDiag | 19 | case tumour Dice (mean) | 0.788 | 0.788 | 0.798 |
|  |  | voxel Dice (pooled) | 0.863 | 0.859 | 0.871 |
|  |  | voxel recall (pooled) | 0.854 | 0.851 | 0.871 |
|  |  | voxel precision (pooled) | 0.872 | 0.868 | 0.871 |
|  |  | lesion recall = sensitivity | 0.960 | 0.960 | 0.960 |
|  |  | lesion precision | 0.750 | 0.774 | 0.706 |
|  |  | lesion F1 | 0.842 | 0.857 | 0.814 |
|  |  | false positives per case | 0.42 | 0.37 | 0.53 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.642 | 0.614 | 0.601 |
|  |  | voxel Dice (pooled) | 0.797 | 0.751 | 0.780 |
|  |  | voxel recall (pooled) | 0.742 | 0.734 | 0.752 |
|  |  | voxel precision (pooled) | 0.861 | 0.770 | 0.811 |
|  |  | lesion recall = sensitivity | 0.963 | 0.926 | 0.926 |
|  |  | lesion precision | 0.553 | 0.446 | 0.532 |
|  |  | lesion F1 | 0.703 | 0.602 | 0.676 |
|  |  | false positives per case | 0.62 | 0.91 | 0.65 |
| WAW-TACE | 14 | case tumour Dice (mean) | 0.747 | 0.757 | 0.748 |
|  |  | voxel Dice (pooled) | 0.788 | 0.817 | 0.825 |
|  |  | voxel recall (pooled) | 0.721 | 0.729 | 0.755 |
|  |  | voxel precision (pooled) | 0.868 | 0.929 | 0.910 |
|  |  | lesion recall = sensitivity | 1.000 | 1.000 | 1.000 |
|  |  | lesion precision | 0.611 | 0.786 | 0.611 |
|  |  | lesion F1 | 0.759 | 0.880 | 0.759 |
|  |  | false positives per case | 1.00 | 0.43 | 1.00 |

### Per size bin

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 18 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 10 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| > 15 mm | 74 | nnU-Net 300 ep | 0.973 [0.91, 0.99] | 0.889 | 0.757 | 0.727 | 9 |
| > 15 mm | 74 | Plain 3D U-Net 300 ep | 0.959 [0.89, 0.99] | 0.910 | 0.753 | 0.726 | 7 |
| > 15 mm | 74 | PLAN stage 1 (seg branch) 300 ep | 0.959 [0.89, 0.99] | 0.835 | 0.773 | 0.727 | 14 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | n/a | n/a | n/a | 0 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| > 15 mm | 25 | nnU-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.854 | 0.772 | 0 |
| > 15 mm | 25 | Plain 3D U-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.851 | 0.776 | 0 |
| > 15 mm | 25 | PLAN stage 1 (seg branch) 300 ep | 0.960 [0.80, 0.99] | 0.960 | 0.871 | 0.784 | 1 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| > 15 mm | 27 | nnU-Net 300 ep | 0.963 [0.82, 0.99] | 0.788 | 0.742 | 0.694 | 7 |
| > 15 mm | 27 | Plain 3D U-Net 300 ep | 0.926 [0.77, 0.98] | 0.781 | 0.734 | 0.681 | 7 |
| > 15 mm | 27 | PLAN stage 1 (seg branch) 300 ep | 0.926 [0.77, 0.98] | 0.714 | 0.752 | 0.677 | 10 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| > 15 mm | 22 | nnU-Net 300 ep | 1.000 [0.85, 1.00] | 0.917 | 0.721 | 0.716 | 2 |
| > 15 mm | 22 | Plain 3D U-Net 300 ep | 1.000 [0.85, 1.00] | 1.000 | 0.729 | 0.724 | 0 |
| > 15 mm | 22 | PLAN stage 1 (seg branch) 300 ep | 1.000 [0.85, 1.00] | 0.880 | 0.755 | 0.722 | 3 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
PLAN stage 1 (seg branch) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.4165, "final_val_loss": -1.2835, "final_global_dice_tumour_liver": [0.8185, 0.9638], "final_online_lesion_sens_prec": [0.871, 0.4426], "training_hours": 21.7}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [183, 218, 246], "current_spacing": [1.0, 1.0, 1.0], "do_dummy_2D_data_aug": false, "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Artefacts

`results/2026-09-28_plan_stage1_segmentation_branch/`: nnunet_cohorts.json, nnunet_nnUNetPlans.json, nnunet_test_per_lesion.csv, nnunet_test_size_binned_metrics.csv, nnunet_training_log_2026_9_27_02_50_22.txt, nnunet_val_per_lesion.csv, nnunet_val_size_binned_metrics.csv, plan1_jobs.json, plan1_plans_summary.json, plan1_test_per_lesion.csv, plan1_test_size_binned_metrics.csv, plan1_training_log_2026_9_27_22_13_27.txt, plan1_training_log_2026_9_28_19_52_43.txt, plan1_training_log_2026_9_28_20_00_48.txt, plan1_val_per_lesion.csv, plan1_val_size_binned_metrics.csv, unet3d_config.json, unet3d_history.json, unet3d_test_per_lesion.csv, unet3d_test_size_binned_metrics.csv, unet3d_val_per_lesion.csv, unet3d_val_size_binned_metrics.csv

