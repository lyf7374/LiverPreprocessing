# 2026-10-04: plan stratified stage1

**Purpose.** PLAN under the reframed protocol, stage 1 of the modified reproduction: PLAN's pixel branch (nnU-Net v1 encoder + FPN decoder + segmentation head) trained on the stratified split with real small lesions and with the lesion-balanced foreground sampling that helped the U-Net. Compared with the U-Net that uses the same split and the same sampling (strat_lesion). Stage 2 (original heads) and stage 2 modified (finer lesion-branch levels, 100 queries) are training from this checkpoint.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: stratified seed 0 (train 727 / val 105 / test 210; lesions of every size in all subsets); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

New nnU-Net v1 task Task521_HCC_strat built from the stratified split (813 training + validation cases after dropping the 19 with an unusable phase, 210 test cases; labels 1 tumour / 2 liver; images identical to the baselines). PLAN's DataLoader3D_Inst was extended with a per-lesion foreground draw: a lesion index (plan_lesion_index.py, 2,231 components, stored in a subfolder because nnU-Net v1 treats every npz of the stage folder as a case) gives 64 sample voxels per component; with cfg.data_loader.lesion_sampling the foreground patch is centred on a voxel of a lesion drawn uniformly among the components with >= 10 voxels (smaller ones are PLAN's ignore label). Otherwise cfg_hcc_plan1 as before: components ('seg',), FPN with 64-channel mask embeddings, RAdam 1e-4, 300 epochs x 250 iterations, patch 112 x 128 x 160, batch 2. Inference: sliding window with mirroring, PLAN post-processing (largest liver component, lesions farther than 3 mm from the liver removed, lesion volume >= 25 mm3), softmax saved (validation.save_npz) and converted to tumour probability maps (plan_npz_to_prob.py) for the FROC; FROC also with the 33-voxel minimum component size used for the U-Net runs. Training 23.4 h (4.7 min per epoch), inference + evaluation 1.3 h (job 30029583).

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `strat_lesion` = U-Net, lesion-balanced sampling (stratified split); `plan1_strat` = PLAN stage 1 (seg branch), lesion sampling, stratified

**Findings.**

Argmax operating points on the stratified test subset (PLAN post-processed vs U-Net with lesion sampling): 5-15 mm sensitivity 0.519 [0.43, 0.60] vs 0.442 [0.36, 0.53]; 5-10 mm 0.344 vs 0.246; 10-15 mm 0.676 vs 0.618; > 15 mm 0.962 vs 0.934; false positives per case 0.93 vs 0.87; lesion recall 0.692 vs 0.663, precision 0.643 vs 0.649, F1 0.667 vs 0.639; case tumour Dice 0.728 vs 0.696. Without PLAN's post-processing the same predictions give 1.89 false positives per case and precision 0.473 at unchanged sensitivity, so half of the argmax advantage is post-processing. Like-for-like FROC with the same 33-voxel component filter (5-15 mm sensitivity at 0.5 / 1 / 2 / 4 FP per case): PLAN stage 1 0.363 / 0.512 / 0.581 / 0.625 versus U-Net 0.349 / 0.465 / 0.554 / n.a.; 5-10 mm at 1 / 2 FP per case 0.328 / 0.426 versus 0.279 / 0.396; on the validation subset PLAN 0.399 / 0.524 at 1 / 2 FP per case. Without any filter the two raw curves coincide (0.421 / 0.527 vs 0.447 / 0.533). Interpretation: with equal sampling and equal post-processing, PLAN's pixel branch is about 5 points more sensitive than the plain U-Net on 5-15 mm lesions at 1 FP per case and 3 points at 2 FP per case, with the gain concentrated in the 5-10 mm bin; the FPN decoder with 64-channel mask features plus RAdam is a slightly better pixel backbone for small lesions than the plain decoder, and PLAN's liver-anchored post-processing is worth about one false positive per case. This is the new best single-stage result (5-15 mm 0.512 at 1 FP per case, 0.581 at 2). Stage 2 variants will show whether the lesion branch adds detection sensitivity or only precision.

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | U-Net, lesion-balanced sampling (stratified split) | PLAN stage 1 (seg branch), lesion sampling, stratified |
|---|---:|---|---:|---:|
| overall | 210 | case tumour Dice (mean) | 0.696 | 0.728 |
|  |  | voxel Dice (pooled) | 0.853 | 0.853 |
|  |  | voxel recall (pooled) | 0.842 | 0.839 |
|  |  | voxel precision (pooled) | 0.864 | 0.868 |
|  |  | lesion recall = sensitivity | 0.663 | 0.692 |
|  |  | lesion precision | 0.649 | 0.643 |
|  |  | lesion F1 | 0.656 | 0.667 |
|  |  | false positives per case | 0.87 | 0.93 |
| MCT-LTDiag | 106 | case tumour Dice (mean) | 0.755 | 0.770 |
|  |  | voxel Dice (pooled) | 0.894 | 0.886 |
|  |  | voxel recall (pooled) | 0.893 | 0.883 |
|  |  | voxel precision (pooled) | 0.895 | 0.889 |
|  |  | lesion recall = sensitivity | 0.593 | 0.627 |
|  |  | lesion precision | 0.763 | 0.723 |
|  |  | lesion F1 | 0.668 | 0.671 |
|  |  | false positives per case | 0.68 | 0.89 |
| PLC-CECT | 69 | case tumour Dice (mean) | 0.589 | 0.648 |
|  |  | voxel Dice (pooled) | 0.808 | 0.811 |
|  |  | voxel recall (pooled) | 0.758 | 0.760 |
|  |  | voxel precision (pooled) | 0.866 | 0.869 |
|  |  | lesion recall = sensitivity | 0.921 | 0.921 |
|  |  | lesion precision | 0.457 | 0.500 |
|  |  | lesion F1 | 0.611 | 0.648 |
|  |  | false positives per case | 1.00 | 0.84 |
| WAW-TACE | 35 | case tumour Dice (mean) | 0.706 | 0.730 |
|  |  | voxel Dice (pooled) | 0.761 | 0.798 |
|  |  | voxel recall (pooled) | 0.843 | 0.858 |
|  |  | voxel precision (pooled) | 0.694 | 0.746 |
|  |  | lesion recall = sensitivity | 0.868 | 0.906 |
|  |  | lesion precision | 0.529 | 0.527 |
|  |  | lesion F1 | 0.657 | 0.667 |
|  |  | false positives per case | 1.17 | 1.23 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 129 | U-Net, lesion-balanced sampling (stratified split) | 0.442 [0.36, 0.53] | 0.373 | 0.358 | 0.212 | 96 |
| **5-15 mm** | 129 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.519 [0.43, 0.60] | 0.344 | 0.399 | 0.255 | 128 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion-balanced sampling (stratified split) | 0.121 [0.07, 0.20] | 0.175 | 0.067 | 0.005 | 52 |
| 0-5 mm (exploratory) | 91 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.088 [0.05, 0.16] | 0.211 | 0.058 | 0.001 | 30 |
| 5-10 mm | 61 | U-Net, lesion-balanced sampling (stratified split) | 0.246 [0.16, 0.37] | 0.192 | 0.152 | 0.056 | 63 |
| 5-10 mm | 61 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.344 [0.24, 0.47] | 0.178 | 0.255 | 0.124 | 97 |
| 10-15 mm | 68 | U-Net, lesion-balanced sampling (stratified split) | 0.618 [0.50, 0.72] | 0.560 | 0.402 | 0.351 | 33 |
| 10-15 mm | 68 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.676 [0.56, 0.78] | 0.597 | 0.429 | 0.372 | 31 |
| > 15 mm | 287 | U-Net, lesion-balanced sampling (stratified split) | 0.934 [0.90, 0.96] | 0.887 | 0.844 | 0.708 | 34 |
| > 15 mm | 287 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.962 [0.93, 0.98] | 0.882 | 0.840 | 0.732 | 37 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 118 | U-Net, lesion-balanced sampling (stratified split) | 0.441 [0.35, 0.53] | 0.571 | 0.351 | 0.207 | 39 |
| **5-15 mm** | 118 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.525 [0.44, 0.61] | 0.484 | 0.398 | 0.253 | 66 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion-balanced sampling (stratified split) | 0.121 [0.07, 0.20] | 0.367 | 0.067 | 0.005 | 19 |
| 0-5 mm (exploratory) | 91 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.088 [0.05, 0.16] | 0.308 | 0.058 | 0.001 | 18 |
| 5-10 mm | 57 | U-Net, lesion-balanced sampling (stratified split) | 0.263 [0.17, 0.39] | 0.326 | 0.164 | 0.060 | 31 |
| 5-10 mm | 57 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.368 [0.26, 0.50] | 0.284 | 0.274 | 0.132 | 53 |
| 10-15 mm | 61 | U-Net, lesion-balanced sampling (stratified split) | 0.607 [0.48, 0.72] | 0.822 | 0.392 | 0.344 | 8 |
| 10-15 mm | 61 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.672 [0.55, 0.78] | 0.759 | 0.425 | 0.367 | 13 |
| > 15 mm | 182 | U-Net, lesion-balanced sampling (stratified split) | 0.929 [0.88, 0.96] | 0.923 | 0.895 | 0.706 | 14 |
| > 15 mm | 182 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.962 [0.92, 0.98] | 0.946 | 0.885 | 0.736 | 10 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 5 | U-Net, lesion-balanced sampling (stratified split) | 0.600 [0.23, 0.88] | 0.071 | 0.487 | 0.364 | 39 |
| **5-15 mm** | 5 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.600 [0.23, 0.88] | 0.077 | 0.380 | 0.354 | 36 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 19 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 9 |
| 5-10 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 24 |
| 5-10 mm | 2 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 25 |
| 10-15 mm | 3 | U-Net, lesion-balanced sampling (stratified split) | 1.000 [0.44, 1.00] | 0.167 | 0.584 | 0.607 | 15 |
| 10-15 mm | 3 | PLAN stage 1 (seg branch), lesion sampling, stratified | 1.000 [0.44, 1.00] | 0.214 | 0.455 | 0.591 | 11 |
| > 15 mm | 58 | U-Net, lesion-balanced sampling (stratified split) | 0.948 [0.86, 0.98] | 0.833 | 0.758 | 0.691 | 11 |
| > 15 mm | 58 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.948 [0.86, 0.98] | 0.809 | 0.760 | 0.696 | 13 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 6 | U-Net, lesion-balanced sampling (stratified split) | 0.333 [0.10, 0.70] | 0.100 | 0.384 | 0.175 | 18 |
| **5-15 mm** | 6 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.333 [0.10, 0.70] | 0.071 | 0.418 | 0.193 | 26 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 14 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 8 |
| 5-10 mm | 2 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 19 |
| 10-15 mm | 4 | U-Net, lesion-balanced sampling (stratified split) | 0.500 [0.15, 0.85] | 0.167 | 0.425 | 0.262 | 10 |
| 10-15 mm | 4 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.500 [0.15, 0.85] | 0.222 | 0.464 | 0.289 | 7 |
| > 15 mm | 47 | U-Net, lesion-balanced sampling (stratified split) | 0.936 [0.83, 0.98] | 0.830 | 0.843 | 0.737 | 9 |
| > 15 mm | 47 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.979 [0.89, 1.00] | 0.767 | 0.859 | 0.762 | 14 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.335 | 0.447 | 0.533 | 0.475 |
| 5-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.421 | 0.527 | 0.539 |
| 5-10 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.100 | 0.257 | 0.369 | 0.290 |
| 5-10 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.218 | 0.361 | 0.371 |
| 10-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.546 | 0.618 | 0.680 | 0.642 |
| 10-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.603 | 0.676 | 0.689 |
| > 15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.890 | 0.938 | 0.965 | 0.941 |
| > 15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.938 | 0.976 | 0.969 |
| all sizes | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.601 | 0.667 | 0.711 | 0.679 |
| all sizes | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.648 | 0.708 | 0.710 |
| 0-5 mm (exploratory) | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.066 | 0.121 | 0.159 | 0.142 |
| 0-5 mm (exploratory) | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.055 | 0.121 | 0.136 |

### Training

```
U-Net, lesion-balanced sampling (stratified split): {"epochs_logged": 300, "mean_epoch_seconds": 49.1, "final_val_patch_dice_liver_tumour": [0.9582, 0.8351], "best_val_patch_dice_tumour": 0.8921, "final_train_loss": 0.3521, "synthetic_lesions_per_epoch_mean": 0.0}
PLAN stage 1 (seg branch), lesion sampling, stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.3986, "final_val_loss": -1.3575, "final_global_dice_tumour_liver": [0.8684, 0.9586], "final_online_lesion_sens_prec": [0.6021, 0.4528], "training_hours": 23.4}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [193, 214, 246], "current_spacing": [1.0, 1.0, 1.0], "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | U-Net, lesion-balanced sampling (stratified split) | PLAN stage 1 (seg branch), lesion sampling, stratified |
|---|---:|---|---:|---:|
| overall | 105 | case tumour Dice (mean) | 0.709 | 0.726 |
|  |  | voxel Dice (pooled) | 0.850 | 0.854 |
|  |  | voxel recall (pooled) | 0.866 | 0.866 |
|  |  | voxel precision (pooled) | 0.835 | 0.842 |
|  |  | lesion recall = sensitivity | 0.577 | 0.620 |
|  |  | lesion precision | 0.707 | 0.663 |
|  |  | lesion F1 | 0.635 | 0.641 |
|  |  | false positives per case | 0.70 | 0.91 |
| MCT-LTDiag | 54 | case tumour Dice (mean) | 0.763 | 0.770 |
|  |  | voxel Dice (pooled) | 0.872 | 0.870 |
|  |  | voxel recall (pooled) | 0.870 | 0.872 |
|  |  | voxel precision (pooled) | 0.874 | 0.868 |
|  |  | lesion recall = sensitivity | 0.510 | 0.557 |
|  |  | lesion precision | 0.777 | 0.746 |
|  |  | lesion F1 | 0.616 | 0.638 |
|  |  | false positives per case | 0.69 | 0.89 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.642 | 0.673 |
|  |  | voxel Dice (pooled) | 0.829 | 0.840 |
|  |  | voxel recall (pooled) | 0.869 | 0.870 |
|  |  | voxel precision (pooled) | 0.792 | 0.812 |
|  |  | lesion recall = sensitivity | 0.964 | 0.964 |
|  |  | lesion precision | 0.562 | 0.519 |
|  |  | lesion F1 | 0.711 | 0.675 |
|  |  | false positives per case | 0.62 | 0.74 |
| WAW-TACE | 17 | case tumour Dice (mean) | 0.650 | 0.676 |
|  |  | voxel Dice (pooled) | 0.837 | 0.835 |
|  |  | voxel recall (pooled) | 0.848 | 0.835 |
|  |  | voxel precision (pooled) | 0.826 | 0.836 |
|  |  | lesion recall = sensitivity | 0.833 | 0.875 |
|  |  | lesion precision | 0.571 | 0.477 |
|  |  | lesion F1 | 0.678 | 0.618 |
|  |  | false positives per case | 0.88 | 1.35 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 98 | U-Net, lesion-balanced sampling (stratified split) | 0.316 [0.23, 0.41] | 0.437 | 0.319 | 0.174 | 40 |
| **5-15 mm** | 98 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.398 [0.31, 0.50] | 0.375 | 0.393 | 0.235 | 65 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion-balanced sampling (stratified split) | 0.094 [0.04, 0.20] | 0.185 | 0.053 | 0.006 | 22 |
| 0-5 mm (exploratory) | 53 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.094 [0.04, 0.20] | 0.185 | 0.095 | 0.010 | 22 |
| 5-10 mm | 58 | U-Net, lesion-balanced sampling (stratified split) | 0.155 [0.08, 0.27] | 0.250 | 0.141 | 0.085 | 27 |
| 5-10 mm | 58 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.259 [0.16, 0.38] | 0.238 | 0.209 | 0.145 | 48 |
| 10-15 mm | 40 | U-Net, lesion-balanced sampling (stratified split) | 0.550 [0.40, 0.69] | 0.629 | 0.373 | 0.303 | 13 |
| 10-15 mm | 40 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.600 [0.45, 0.74] | 0.585 | 0.449 | 0.367 | 17 |
| > 15 mm | 154 | U-Net, lesion-balanced sampling (stratified split) | 0.909 [0.85, 0.95] | 0.927 | 0.868 | 0.681 | 11 |
| > 15 mm | 154 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.942 [0.89, 0.97] | 0.942 | 0.868 | 0.711 | 9 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 94 | U-Net, lesion-balanced sampling (stratified split) | 0.319 [0.23, 0.42] | 0.556 | 0.317 | 0.174 | 24 |
| **5-15 mm** | 94 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.394 [0.30, 0.49] | 0.536 | 0.388 | 0.233 | 32 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion-balanced sampling (stratified split) | 0.094 [0.04, 0.20] | 0.294 | 0.053 | 0.006 | 12 |
| 0-5 mm (exploratory) | 53 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.094 [0.04, 0.20] | 0.263 | 0.095 | 0.010 | 14 |
| 5-10 mm | 57 | U-Net, lesion-balanced sampling (stratified split) | 0.158 [0.09, 0.27] | 0.300 | 0.144 | 0.086 | 21 |
| 5-10 mm | 57 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.263 [0.17, 0.39] | 0.395 | 0.214 | 0.147 | 23 |
| 10-15 mm | 37 | U-Net, lesion-balanced sampling (stratified split) | 0.568 [0.41, 0.71] | 0.875 | 0.372 | 0.309 | 3 |
| 10-15 mm | 37 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.595 [0.43, 0.74] | 0.710 | 0.444 | 0.366 | 9 |
| > 15 mm | 106 | U-Net, lesion-balanced sampling (stratified split) | 0.887 [0.81, 0.93] | 0.989 | 0.874 | 0.659 | 1 |
| > 15 mm | 106 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.934 [0.87, 0.97] | 0.980 | 0.876 | 0.692 | 2 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 9 |
| **5-15 mm** | 1 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 17 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 1 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| > 15 mm | 27 | U-Net, lesion-balanced sampling (stratified split) | 1.000 [0.88, 1.00] | 0.844 | 0.869 | 0.776 | 5 |
| > 15 mm | 27 | PLAN stage 1 (seg branch), lesion sampling, stratified | 1.000 [0.88, 1.00] | 0.844 | 0.870 | 0.781 | 5 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 3 | U-Net, lesion-balanced sampling (stratified split) | 0.333 [0.06, 0.79] | 0.125 | 0.416 | 0.231 | 7 |
| **5-15 mm** | 3 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.667 [0.21, 0.94] | 0.111 | 0.563 | 0.373 | 16 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 1 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 13 |
| 10-15 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.500 [0.09, 0.91] | 0.167 | 0.458 | 0.347 | 5 |
| 10-15 mm | 2 | PLAN stage 1 (seg branch), lesion sampling, stratified | 1.000 [0.34, 1.00] | 0.400 | 0.619 | 0.560 | 3 |
| > 15 mm | 21 | U-Net, lesion-balanced sampling (stratified split) | 0.905 [0.71, 0.97] | 0.792 | 0.849 | 0.671 | 5 |
| > 15 mm | 21 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.905 [0.71, 0.97] | 0.905 | 0.836 | 0.719 | 2 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.260 | 0.332 | 0.394 | 0.375 |
| 5-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.377 | 0.423 | 0.485 |
| 5-10 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.121 | 0.182 | 0.234 | 0.231 |
| 5-10 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.249 | 0.283 | 0.373 |
| 10-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.463 | 0.550 | 0.625 | 0.584 |
| 10-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.561 | 0.625 | 0.647 |
| > 15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.896 | 0.913 | 0.943 | 0.932 |
| > 15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.929 | 0.953 | 0.958 |
| all sizes | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.548 | 0.584 | 0.623 | 0.612 |
| all sizes | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.605 | 0.642 | 0.674 |
| 0-5 mm (exploratory) | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.066 | 0.094 | 0.113 | 0.123 |
| 0-5 mm (exploratory) | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.084 | 0.145 | 0.201 |

### Training

```
U-Net, lesion-balanced sampling (stratified split): {"epochs_logged": 300, "mean_epoch_seconds": 49.1, "final_val_patch_dice_liver_tumour": [0.9582, 0.8351], "best_val_patch_dice_tumour": 0.8921, "final_train_loss": 0.3521, "synthetic_lesions_per_epoch_mean": 0.0}
PLAN stage 1 (seg branch), lesion sampling, stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.3986, "final_val_loss": -1.3575, "final_global_dice_tumour_liver": [0.8684, 0.9586], "final_online_lesion_sens_prec": [0.6021, 0.4528], "training_hours": 23.4}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [193, 214, 246], "current_spacing": [1.0, 1.0, 1.0], "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Artefacts

`results/2026-10-04_plan_stratified_stage1/`: plan1_strat_jobs.json, plan1_strat_plans_summary.json, plan1_strat_test_froc.csv, plan1_strat_test_froc_min33_froc_summary.json, plan1_strat_test_froc_summary.json, plan1_strat_test_per_lesion.csv, plan1_strat_test_size_binned_metrics.csv, plan1_strat_training_log_2026_10_3_11_19_13.txt, plan1_strat_training_log_2026_10_4_10_45_29.txt, plan1_strat_training_log_2026_10_4_10_58_42.txt, plan1_strat_val_froc.csv, plan1_strat_val_froc_min33_froc_summary.json, plan1_strat_val_froc_summary.json, plan1_strat_val_per_lesion.csv, plan1_strat_val_size_binned_metrics.csv, strat_lesion_config.json, strat_lesion_history.json, strat_lesion_jobs.json, strat_lesion_test_evaluation_minsize100_size_binned_metrics.csv, strat_lesion_test_froc.csv, strat_lesion_test_froc_min33_froc_summary.json, strat_lesion_test_froc_summary.json, strat_lesion_test_per_lesion.csv, strat_lesion_test_size_binned_metrics.csv, strat_lesion_val_froc.csv, strat_lesion_val_froc_min33_froc_summary.json, strat_lesion_val_froc_summary.json, strat_lesion_val_per_lesion.csv, strat_lesion_val_size_binned_metrics.csv

