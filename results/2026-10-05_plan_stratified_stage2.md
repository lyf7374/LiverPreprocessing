# 2026-10-05: plan stratified stage2

**Purpose.** PLAN stage 2 on the stratified split: the full Pixel-Lesion-pAtient Network (pixel + lesion + patient branches) initialised from the stage-1 checkpoint trained with lesion-balanced sampling, in two versions: the original heads (cfg_hcc_plan2) and a version modified for small lesions (cfg_hcc_plan2_mod: the lesion branch attends to decoder levels 1/8, 1/4 and 1/2 instead of 1/16, 1/8 and 1/4, and uses 100 instead of 50 queries). Does the lesion branch add detection sensitivity for 5-15 mm lesions or only precision, and do the finer levels help? Compared with PLAN stage 1 and the lesion-sampling U-Net on the same split.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: stratified seed 0 (train 727 / val 105 / test 210; lesions of every size in all subsets); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

Both stage-2 runs: Task521_HCC_strat (813 training + validation cases, 210 test), lesion-balanced sampling (plan_lesion_index.py), components ('cls', 'det', 'seg'), U-Net + FPN weights loaded from hcc_plan1_strat, RAdam 1e-4, 300 epochs x 250 iterations, patch 112 x 128 x 160, batch 2; lesion branch = 3-layer masked-attention transformer decoder (Mask2Former losses: no-object 0.1, dice 5, mask 5, CE 2, 12,544 sampled points, foreground-enhanced sampling 2e-4), patient branch (60 patches, one label: tumour present) and det-cls consistency loss from epoch 50. Original: det levels [-5, -4, -3] (1/16, 1/8, 1/4), 50 queries, 23.0 h (job 30029584). Modified: det levels [-4, -3, -2] (1/8, 1/4, 1/2), 100 queries, 22.7 h (job 30029585); same memory footprint class (one A100 40 GB). Inference as in PLAN: the lesion branch's masks replace the pixel softmax (det_head.inference), mirroring, PLAN post-processing (largest liver component, lesions > 3 mm from the liver removed, >= 25 mm3); the saved softmax (per-lesion mask scores) gives the FROC, computed raw and with the 33-voxel minimum component size used for the U-Net runs (froc_min33; CPU jobs 30459672 / 30459673).

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `strat_lesion` = U-Net, lesion-balanced sampling (stratified split); `plan1_strat` = PLAN stage 1 (seg branch), lesion sampling, stratified; `plan2_strat` = PLAN stage 2 (seg + det + cls), stratified; `plan2mod_strat` = PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified

**Findings.**

Argmax operating points (test, 210 cases; U-Net lesion sampling / PLAN stage 1 / stage 2 / stage 2 modified): 5-15 mm sensitivity 0.442 / 0.519 / 0.473 [0.39, 0.56] / 0.481 [0.40, 0.57]; 5-10 mm 0.246 / 0.344 / 0.295 / 0.279; 10-15 mm 0.618 / 0.676 / 0.632 / 0.662; > 15 mm 0.934 / 0.962 / 0.951 / 0.951; false positives per case 0.87 / 0.93 / 0.63 / 0.65; lesion precision 0.649 / 0.643 / 0.719 / 0.713; F1 0.639 / 0.667 / 0.694 / 0.692; case tumour Dice 0.696 / 0.728 / 0.740 / 0.743. The lesion branch therefore moves the default operating point towards precision (0.3 fewer false positives per case, best Dice and F1) at the cost of 4 points of 5-15 mm sensitivity. The curves tell a different story. Like-for-like FROC with the 33-voxel component filter, 5-15 mm sensitivity at 0.5 / 1 / 2 / 4 FP per case: U-Net 0.349 / 0.465 / 0.554 / n.a.; stage 1 0.363 / 0.512 / 0.581 / 0.625; stage 2 0.422 / 0.519 / 0.592 / 0.636; stage 2 modified 0.451 / 0.531 / 0.592 / 0.643. 5-10 mm at 1 / 2 FP per case: 0.279 / 0.396, 0.328 / 0.426, 0.344 / 0.421, 0.361 / 0.391. Validation subset (105 cases, 98 lesions of 5-15 mm) at 0.5 / 1 / 2 / 4 FP per case: U-Net 0.315 / 0.363 / 0.423 / n.a.; stage 1 0.327 / 0.399 / 0.524 / 0.598; stage 2 0.305 / 0.374 / 0.484 / 0.585; stage 2 modified 0.348 / 0.435 / 0.503 / 0.569. The lesion-branch outputs also extend the operating range to 0.07 FP per case (stage 1 stops at 0.28), because the query scores suppress most background components. Without the size filter (raw FROC) the original stage 2 loses sensitivity at 1 FP per case (0.369 vs stage 1 0.421) and the modified one does not (0.460); the filter removes that difference, so the original heads produce many tiny confident components that the finer levels avoid. Interpretation: (1) the modified stage 2 is the best model at <= 1 FP per case on both test (0.451 / 0.531) and validation (0.348 / 0.435), i.e. +9 / +7 points over the U-Net and +9 / +2 over stage 1 at 0.5 / 1 FP per case on test; (2) above 2 FP per case all PLAN variants converge (0.59-0.64) and stage 1 is as good; (3) the finer attention levels and more queries help mainly at low false-positive rates and in the 5-10 mm bin (0.361 vs 0.344 at 1 FP per case), confirming that the lesion branch was limited by its 1/4 scale; (4) the lesion branch's own post-processing-free FROC floor (0.07 FP per case) makes it the natural high-precision operating mode, while the pixel branch (stage 1) remains the better high-recall candidate generator. Recommended configuration for the internal fine-tuning: PLAN with lesion sampling, modified lesion branch, 33-voxel filter; report the 5-15 mm FROC at 0.5, 1 and 2 FP per case.

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | U-Net, lesion-balanced sampling (stratified split) | PLAN stage 1 (seg branch), lesion sampling, stratified | PLAN stage 2 (seg + det + cls), stratified | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified |
|---|---:|---|---:|---:|---:|---:|
| overall | 210 | case tumour Dice (mean) | 0.696 | 0.728 | 0.740 | 0.743 |
|  |  | voxel Dice (pooled) | 0.853 | 0.853 | 0.851 | 0.845 |
|  |  | voxel recall (pooled) | 0.842 | 0.839 | 0.829 | 0.822 |
|  |  | voxel precision (pooled) | 0.864 | 0.868 | 0.874 | 0.870 |
|  |  | lesion recall = sensitivity | 0.663 | 0.692 | 0.671 | 0.673 |
|  |  | lesion precision | 0.649 | 0.643 | 0.719 | 0.713 |
|  |  | lesion F1 | 0.656 | 0.667 | 0.694 | 0.692 |
|  |  | false positives per case | 0.87 | 0.93 | 0.63 | 0.65 |
| MCT-LTDiag | 106 | case tumour Dice (mean) | 0.755 | 0.770 | 0.777 | 0.778 |
|  |  | voxel Dice (pooled) | 0.894 | 0.886 | 0.884 | 0.878 |
|  |  | voxel recall (pooled) | 0.893 | 0.883 | 0.867 | 0.858 |
|  |  | voxel precision (pooled) | 0.895 | 0.889 | 0.901 | 0.900 |
|  |  | lesion recall = sensitivity | 0.593 | 0.627 | 0.604 | 0.606 |
|  |  | lesion precision | 0.763 | 0.723 | 0.789 | 0.777 |
|  |  | lesion F1 | 0.668 | 0.671 | 0.684 | 0.681 |
|  |  | false positives per case | 0.68 | 0.89 | 0.59 | 0.64 |
| PLC-CECT | 69 | case tumour Dice (mean) | 0.589 | 0.648 | 0.679 | 0.692 |
|  |  | voxel Dice (pooled) | 0.808 | 0.811 | 0.814 | 0.812 |
|  |  | voxel recall (pooled) | 0.758 | 0.760 | 0.758 | 0.756 |
|  |  | voxel precision (pooled) | 0.866 | 0.869 | 0.879 | 0.877 |
|  |  | lesion recall = sensitivity | 0.921 | 0.921 | 0.889 | 0.889 |
|  |  | lesion precision | 0.457 | 0.500 | 0.629 | 0.636 |
|  |  | lesion F1 | 0.611 | 0.648 | 0.737 | 0.742 |
|  |  | false positives per case | 1.00 | 0.84 | 0.48 | 0.46 |
| WAW-TACE | 35 | case tumour Dice (mean) | 0.706 | 0.730 | 0.721 | 0.716 |
|  |  | voxel Dice (pooled) | 0.761 | 0.798 | 0.779 | 0.766 |
|  |  | voxel recall (pooled) | 0.843 | 0.858 | 0.851 | 0.847 |
|  |  | voxel precision (pooled) | 0.694 | 0.746 | 0.718 | 0.699 |
|  |  | lesion recall = sensitivity | 0.868 | 0.906 | 0.906 | 0.906 |
|  |  | lesion precision | 0.529 | 0.527 | 0.565 | 0.565 |
|  |  | lesion F1 | 0.657 | 0.667 | 0.696 | 0.696 |
|  |  | false positives per case | 1.17 | 1.23 | 1.06 | 1.06 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 129 | U-Net, lesion-balanced sampling (stratified split) | 0.442 [0.36, 0.53] | 0.373 | 0.358 | 0.212 | 96 |
| **5-15 mm** | 129 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.519 [0.43, 0.60] | 0.344 | 0.399 | 0.255 | 128 |
| **5-15 mm** | 129 | PLAN stage 2 (seg + det + cls), stratified | 0.473 [0.39, 0.56] | 0.427 | 0.417 | 0.257 | 82 |
| **5-15 mm** | 129 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.481 [0.40, 0.57] | 0.425 | 0.453 | 0.267 | 84 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion-balanced sampling (stratified split) | 0.121 [0.07, 0.20] | 0.175 | 0.067 | 0.005 | 52 |
| 0-5 mm (exploratory) | 91 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.088 [0.05, 0.16] | 0.211 | 0.058 | 0.001 | 30 |
| 0-5 mm (exploratory) | 91 | PLAN stage 2 (seg + det + cls), stratified | 0.066 [0.03, 0.14] | 0.353 | 0.053 | 0.001 | 11 |
| 0-5 mm (exploratory) | 91 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.066 [0.03, 0.14] | 0.261 | 0.052 | 0.002 | 17 |
| 5-10 mm | 61 | U-Net, lesion-balanced sampling (stratified split) | 0.246 [0.16, 0.37] | 0.192 | 0.152 | 0.056 | 63 |
| 5-10 mm | 61 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.344 [0.24, 0.47] | 0.178 | 0.255 | 0.124 | 97 |
| 5-10 mm | 61 | PLAN stage 2 (seg + det + cls), stratified | 0.295 [0.20, 0.42] | 0.273 | 0.261 | 0.117 | 48 |
| 5-10 mm | 61 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.279 [0.18, 0.40] | 0.254 | 0.283 | 0.101 | 50 |
| 10-15 mm | 68 | U-Net, lesion-balanced sampling (stratified split) | 0.618 [0.50, 0.72] | 0.560 | 0.402 | 0.351 | 33 |
| 10-15 mm | 68 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.676 [0.56, 0.78] | 0.597 | 0.429 | 0.372 | 31 |
| 10-15 mm | 68 | PLAN stage 2 (seg + det + cls), stratified | 0.632 [0.51, 0.74] | 0.558 | 0.450 | 0.384 | 34 |
| 10-15 mm | 68 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.662 [0.54, 0.76] | 0.570 | 0.489 | 0.415 | 34 |
| > 15 mm | 287 | U-Net, lesion-balanced sampling (stratified split) | 0.934 [0.90, 0.96] | 0.887 | 0.844 | 0.708 | 34 |
| > 15 mm | 287 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.962 [0.93, 0.98] | 0.882 | 0.840 | 0.732 | 37 |
| > 15 mm | 287 | PLAN stage 2 (seg + det + cls), stratified | 0.951 [0.92, 0.97] | 0.872 | 0.830 | 0.744 | 40 |
| > 15 mm | 287 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.951 [0.92, 0.97] | 0.883 | 0.823 | 0.743 | 36 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 118 | U-Net, lesion-balanced sampling (stratified split) | 0.441 [0.35, 0.53] | 0.571 | 0.351 | 0.207 | 39 |
| **5-15 mm** | 118 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.525 [0.44, 0.61] | 0.484 | 0.398 | 0.253 | 66 |
| **5-15 mm** | 118 | PLAN stage 2 (seg + det + cls), stratified | 0.475 [0.39, 0.56] | 0.549 | 0.416 | 0.257 | 46 |
| **5-15 mm** | 118 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.483 [0.39, 0.57] | 0.548 | 0.453 | 0.265 | 47 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion-balanced sampling (stratified split) | 0.121 [0.07, 0.20] | 0.367 | 0.067 | 0.005 | 19 |
| 0-5 mm (exploratory) | 91 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.088 [0.05, 0.16] | 0.308 | 0.058 | 0.001 | 18 |
| 0-5 mm (exploratory) | 91 | PLAN stage 2 (seg + det + cls), stratified | 0.066 [0.03, 0.14] | 0.500 | 0.053 | 0.001 | 6 |
| 0-5 mm (exploratory) | 91 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.066 [0.03, 0.14] | 0.353 | 0.052 | 0.002 | 11 |
| 5-10 mm | 57 | U-Net, lesion-balanced sampling (stratified split) | 0.263 [0.17, 0.39] | 0.326 | 0.164 | 0.060 | 31 |
| 5-10 mm | 57 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.368 [0.26, 0.50] | 0.284 | 0.274 | 0.132 | 53 |
| 5-10 mm | 57 | PLAN stage 2 (seg + det + cls), stratified | 0.298 [0.20, 0.43] | 0.370 | 0.272 | 0.118 | 29 |
| 5-10 mm | 57 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.281 [0.18, 0.41] | 0.348 | 0.286 | 0.096 | 30 |
| 10-15 mm | 61 | U-Net, lesion-balanced sampling (stratified split) | 0.607 [0.48, 0.72] | 0.822 | 0.392 | 0.344 | 8 |
| 10-15 mm | 61 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.672 [0.55, 0.78] | 0.759 | 0.425 | 0.367 | 13 |
| 10-15 mm | 61 | PLAN stage 2 (seg + det + cls), stratified | 0.639 [0.51, 0.75] | 0.696 | 0.447 | 0.387 | 17 |
| 10-15 mm | 61 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.672 [0.55, 0.78] | 0.707 | 0.490 | 0.422 | 17 |
| > 15 mm | 182 | U-Net, lesion-balanced sampling (stratified split) | 0.929 [0.88, 0.96] | 0.923 | 0.895 | 0.706 | 14 |
| > 15 mm | 182 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.962 [0.92, 0.98] | 0.946 | 0.885 | 0.736 | 10 |
| > 15 mm | 182 | PLAN stage 2 (seg + det + cls), stratified | 0.956 [0.92, 0.98] | 0.941 | 0.869 | 0.747 | 11 |
| > 15 mm | 182 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.956 [0.92, 0.98] | 0.946 | 0.859 | 0.751 | 10 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 5 | U-Net, lesion-balanced sampling (stratified split) | 0.600 [0.23, 0.88] | 0.071 | 0.487 | 0.364 | 39 |
| **5-15 mm** | 5 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.600 [0.23, 0.88] | 0.077 | 0.380 | 0.354 | 36 |
| **5-15 mm** | 5 | PLAN stage 2 (seg + det + cls), stratified | 0.600 [0.23, 0.88] | 0.143 | 0.442 | 0.348 | 18 |
| **5-15 mm** | 5 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.600 [0.23, 0.88] | 0.136 | 0.476 | 0.411 | 19 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 19 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 9 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls), stratified | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 24 |
| 5-10 mm | 2 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 25 |
| 5-10 mm | 2 | PLAN stage 2 (seg + det + cls), stratified | 0.500 [0.09, 0.91] | 0.083 | 0.194 | 0.202 | 11 |
| 5-10 mm | 2 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.500 [0.09, 0.91] | 0.091 | 0.440 | 0.357 | 10 |
| 10-15 mm | 3 | U-Net, lesion-balanced sampling (stratified split) | 1.000 [0.44, 1.00] | 0.167 | 0.584 | 0.607 | 15 |
| 10-15 mm | 3 | PLAN stage 1 (seg branch), lesion sampling, stratified | 1.000 [0.44, 1.00] | 0.214 | 0.455 | 0.591 | 11 |
| 10-15 mm | 3 | PLAN stage 2 (seg + det + cls), stratified | 0.667 [0.21, 0.94] | 0.222 | 0.491 | 0.444 | 7 |
| 10-15 mm | 3 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.667 [0.21, 0.94] | 0.182 | 0.483 | 0.446 | 9 |
| > 15 mm | 58 | U-Net, lesion-balanced sampling (stratified split) | 0.948 [0.86, 0.98] | 0.833 | 0.758 | 0.691 | 11 |
| > 15 mm | 58 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.948 [0.86, 0.98] | 0.809 | 0.760 | 0.696 | 13 |
| > 15 mm | 58 | PLAN stage 2 (seg + det + cls), stratified | 0.914 [0.81, 0.96] | 0.815 | 0.758 | 0.703 | 12 |
| > 15 mm | 58 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.914 [0.81, 0.96] | 0.841 | 0.756 | 0.697 | 10 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 6 | U-Net, lesion-balanced sampling (stratified split) | 0.333 [0.10, 0.70] | 0.100 | 0.384 | 0.175 | 18 |
| **5-15 mm** | 6 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.333 [0.10, 0.70] | 0.071 | 0.418 | 0.193 | 26 |
| **5-15 mm** | 6 | PLAN stage 2 (seg + det + cls), stratified | 0.333 [0.10, 0.70] | 0.100 | 0.414 | 0.191 | 18 |
| **5-15 mm** | 6 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.333 [0.10, 0.70] | 0.100 | 0.423 | 0.194 | 18 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 14 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls), stratified | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 8 |
| 5-10 mm | 2 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 19 |
| 5-10 mm | 2 | PLAN stage 2 (seg + det + cls), stratified | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 8 |
| 5-10 mm | 2 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 10 |
| 10-15 mm | 4 | U-Net, lesion-balanced sampling (stratified split) | 0.500 [0.15, 0.85] | 0.167 | 0.425 | 0.262 | 10 |
| 10-15 mm | 4 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.500 [0.15, 0.85] | 0.222 | 0.464 | 0.289 | 7 |
| 10-15 mm | 4 | PLAN stage 2 (seg + det + cls), stratified | 0.500 [0.15, 0.85] | 0.167 | 0.459 | 0.287 | 10 |
| 10-15 mm | 4 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.500 [0.15, 0.85] | 0.200 | 0.469 | 0.290 | 8 |
| > 15 mm | 47 | U-Net, lesion-balanced sampling (stratified split) | 0.936 [0.83, 0.98] | 0.830 | 0.843 | 0.737 | 9 |
| > 15 mm | 47 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.979 [0.89, 1.00] | 0.767 | 0.859 | 0.762 | 14 |
| > 15 mm | 47 | PLAN stage 2 (seg + det + cls), stratified | 0.979 [0.89, 1.00] | 0.730 | 0.851 | 0.783 | 17 |
| > 15 mm | 47 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.979 [0.89, 1.00] | 0.742 | 0.847 | 0.767 | 16 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.335 | 0.447 | 0.533 | 0.475 |
| 5-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.421 | 0.527 | 0.539 |
| 5-15 mm | PLAN stage 2 (seg + det + cls), stratified | 0.151 | 0.226 | 0.369 | 0.519 | 0.406 |
| 5-15 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.174 | 0.238 | 0.460 | 0.531 | 0.435 |
| 5-10 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.100 | 0.257 | 0.369 | 0.290 |
| 5-10 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.218 | 0.361 | 0.371 |
| 5-10 mm | PLAN stage 2 (seg + det + cls), stratified | 0.033 | 0.066 | 0.152 | 0.344 | 0.237 |
| 5-10 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.035 | 0.070 | 0.235 | 0.361 | 0.255 |
| 10-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.546 | 0.618 | 0.680 | 0.642 |
| 10-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.603 | 0.676 | 0.689 |
| 10-15 mm | PLAN stage 2 (seg + det + cls), stratified | 0.256 | 0.369 | 0.563 | 0.676 | 0.557 |
| 10-15 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.298 | 0.389 | 0.662 | 0.684 | 0.597 |
| > 15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.890 | 0.938 | 0.965 | 0.941 |
| > 15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.938 | 0.976 | 0.969 |
| > 15 mm | PLAN stage 2 (seg + det + cls), stratified | 0.792 | 0.889 | 0.938 | 0.965 | 0.923 |
| > 15 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.802 | 0.870 | 0.944 | 0.969 | 0.923 |
| all sizes | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.601 | 0.667 | 0.711 | 0.679 |
| all sizes | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.648 | 0.708 | 0.710 |
| all sizes | PLAN stage 2 (seg + det + cls), stratified | 0.489 | 0.563 | 0.634 | 0.699 | 0.642 |
| all sizes | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.500 | 0.555 | 0.663 | 0.700 | 0.649 |
| 0-5 mm (exploratory) | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.066 | 0.121 | 0.159 | 0.142 |
| 0-5 mm (exploratory) | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.055 | 0.121 | 0.136 |
| 0-5 mm (exploratory) | PLAN stage 2 (seg + det + cls), stratified | 0.011 | 0.011 | 0.055 | 0.115 | 0.092 |
| 0-5 mm (exploratory) | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.011 | 0.011 | 0.066 | 0.093 | 0.089 |

### Training

```
U-Net, lesion-balanced sampling (stratified split): {"epochs_logged": 300, "mean_epoch_seconds": 49.1, "final_val_patch_dice_liver_tumour": [0.9582, 0.8351], "best_val_patch_dice_tumour": 0.8921, "final_train_loss": 0.3521, "synthetic_lesions_per_epoch_mean": 0.0}
PLAN stage 1 (seg branch), lesion sampling, stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.3986, "final_val_loss": -1.3575, "final_global_dice_tumour_liver": [0.8684, 0.9586], "final_online_lesion_sens_prec": [0.6021, 0.4528], "training_hours": 23.4}
PLAN stage 2 (seg + det + cls), stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 5.9154, "final_val_loss": 7.5229, "final_global_dice_tumour_liver": [0.8494, 0.9619], "final_online_lesion_sens_prec": [0.6788, 0.56], "training_hours": 23.0}
PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 5.766, "final_val_loss": 7.5256, "final_global_dice_tumour_liver": [0.855, 0.9587], "final_online_lesion_sens_prec": [0.6236, 0.6687], "training_hours": 22.7}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [193, 214, 246], "current_spacing": [1.0, 1.0, 1.0], "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | U-Net, lesion-balanced sampling (stratified split) | PLAN stage 1 (seg branch), lesion sampling, stratified | PLAN stage 2 (seg + det + cls), stratified | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified |
|---|---:|---|---:|---:|---:|---:|
| overall | 105 | case tumour Dice (mean) | 0.709 | 0.726 | 0.739 | 0.734 |
|  |  | voxel Dice (pooled) | 0.850 | 0.854 | 0.863 | 0.863 |
|  |  | voxel recall (pooled) | 0.866 | 0.866 | 0.864 | 0.867 |
|  |  | voxel precision (pooled) | 0.835 | 0.842 | 0.862 | 0.859 |
|  |  | lesion recall = sensitivity | 0.577 | 0.620 | 0.574 | 0.610 |
|  |  | lesion precision | 0.707 | 0.663 | 0.761 | 0.769 |
|  |  | lesion F1 | 0.635 | 0.641 | 0.654 | 0.680 |
|  |  | false positives per case | 0.70 | 0.91 | 0.52 | 0.53 |
| MCT-LTDiag | 54 | case tumour Dice (mean) | 0.763 | 0.770 | 0.775 | 0.774 |
|  |  | voxel Dice (pooled) | 0.872 | 0.870 | 0.881 | 0.881 |
|  |  | voxel recall (pooled) | 0.870 | 0.872 | 0.880 | 0.885 |
|  |  | voxel precision (pooled) | 0.874 | 0.868 | 0.882 | 0.876 |
|  |  | lesion recall = sensitivity | 0.510 | 0.557 | 0.506 | 0.549 |
|  |  | lesion precision | 0.777 | 0.746 | 0.810 | 0.842 |
|  |  | lesion F1 | 0.616 | 0.638 | 0.623 | 0.665 |
|  |  | false positives per case | 0.69 | 0.89 | 0.56 | 0.48 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.642 | 0.673 | 0.713 | 0.698 |
|  |  | voxel Dice (pooled) | 0.829 | 0.840 | 0.851 | 0.851 |
|  |  | voxel recall (pooled) | 0.869 | 0.870 | 0.861 | 0.862 |
|  |  | voxel precision (pooled) | 0.792 | 0.812 | 0.841 | 0.841 |
|  |  | lesion recall = sensitivity | 0.964 | 0.964 | 0.964 | 0.964 |
|  |  | lesion precision | 0.562 | 0.519 | 0.614 | 0.562 |
|  |  | lesion F1 | 0.711 | 0.675 | 0.750 | 0.711 |
|  |  | false positives per case | 0.62 | 0.74 | 0.50 | 0.62 |
| WAW-TACE | 17 | case tumour Dice (mean) | 0.650 | 0.676 | 0.669 | 0.666 |
|  |  | voxel Dice (pooled) | 0.837 | 0.835 | 0.835 | 0.834 |
|  |  | voxel recall (pooled) | 0.848 | 0.835 | 0.821 | 0.819 |
|  |  | voxel precision (pooled) | 0.826 | 0.836 | 0.849 | 0.850 |
|  |  | lesion recall = sensitivity | 0.833 | 0.875 | 0.833 | 0.833 |
|  |  | lesion precision | 0.571 | 0.477 | 0.714 | 0.690 |
|  |  | lesion F1 | 0.678 | 0.618 | 0.769 | 0.755 |
|  |  | false positives per case | 0.88 | 1.35 | 0.47 | 0.53 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 98 | U-Net, lesion-balanced sampling (stratified split) | 0.316 [0.23, 0.41] | 0.437 | 0.319 | 0.174 | 40 |
| **5-15 mm** | 98 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.398 [0.31, 0.50] | 0.375 | 0.393 | 0.235 | 65 |
| **5-15 mm** | 98 | PLAN stage 2 (seg + det + cls), stratified | 0.296 [0.21, 0.39] | 0.426 | 0.367 | 0.182 | 39 |
| **5-15 mm** | 98 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.357 [0.27, 0.46] | 0.473 | 0.389 | 0.205 | 39 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion-balanced sampling (stratified split) | 0.094 [0.04, 0.20] | 0.185 | 0.053 | 0.006 | 22 |
| 0-5 mm (exploratory) | 53 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.094 [0.04, 0.20] | 0.185 | 0.095 | 0.010 | 22 |
| 0-5 mm (exploratory) | 53 | PLAN stage 2 (seg + det + cls), stratified | 0.057 [0.02, 0.15] | 0.333 | 0.098 | 0.007 | 6 |
| 0-5 mm (exploratory) | 53 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.075 [0.03, 0.18] | 0.500 | 0.117 | 0.005 | 4 |
| 5-10 mm | 58 | U-Net, lesion-balanced sampling (stratified split) | 0.155 [0.08, 0.27] | 0.250 | 0.141 | 0.085 | 27 |
| 5-10 mm | 58 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.259 [0.16, 0.38] | 0.238 | 0.209 | 0.145 | 48 |
| 5-10 mm | 58 | PLAN stage 2 (seg + det + cls), stratified | 0.138 [0.07, 0.25] | 0.267 | 0.141 | 0.071 | 22 |
| 5-10 mm | 58 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.207 [0.12, 0.33] | 0.387 | 0.176 | 0.101 | 19 |
| 10-15 mm | 40 | U-Net, lesion-balanced sampling (stratified split) | 0.550 [0.40, 0.69] | 0.629 | 0.373 | 0.303 | 13 |
| 10-15 mm | 40 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.600 [0.45, 0.74] | 0.585 | 0.449 | 0.367 | 17 |
| 10-15 mm | 40 | PLAN stage 2 (seg + det + cls), stratified | 0.525 [0.37, 0.67] | 0.553 | 0.435 | 0.342 | 17 |
| 10-15 mm | 40 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.575 [0.42, 0.71] | 0.535 | 0.454 | 0.356 | 20 |
| > 15 mm | 154 | U-Net, lesion-balanced sampling (stratified split) | 0.909 [0.85, 0.95] | 0.927 | 0.868 | 0.681 | 11 |
| > 15 mm | 154 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.942 [0.89, 0.97] | 0.942 | 0.868 | 0.711 | 9 |
| > 15 mm | 154 | PLAN stage 2 (seg + det + cls), stratified | 0.929 [0.88, 0.96] | 0.935 | 0.866 | 0.717 | 10 |
| > 15 mm | 154 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.955 [0.91, 0.98] | 0.919 | 0.869 | 0.722 | 13 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 94 | U-Net, lesion-balanced sampling (stratified split) | 0.319 [0.23, 0.42] | 0.556 | 0.317 | 0.174 | 24 |
| **5-15 mm** | 94 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.394 [0.30, 0.49] | 0.536 | 0.388 | 0.233 | 32 |
| **5-15 mm** | 94 | PLAN stage 2 (seg + det + cls), stratified | 0.298 [0.21, 0.40] | 0.538 | 0.369 | 0.182 | 24 |
| **5-15 mm** | 94 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.362 [0.27, 0.46] | 0.630 | 0.393 | 0.206 | 20 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion-balanced sampling (stratified split) | 0.094 [0.04, 0.20] | 0.294 | 0.053 | 0.006 | 12 |
| 0-5 mm (exploratory) | 53 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.094 [0.04, 0.20] | 0.263 | 0.095 | 0.010 | 14 |
| 0-5 mm (exploratory) | 53 | PLAN stage 2 (seg + det + cls), stratified | 0.057 [0.02, 0.15] | 0.429 | 0.098 | 0.007 | 4 |
| 0-5 mm (exploratory) | 53 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.075 [0.03, 0.18] | 0.667 | 0.117 | 0.005 | 2 |
| 5-10 mm | 57 | U-Net, lesion-balanced sampling (stratified split) | 0.158 [0.09, 0.27] | 0.300 | 0.144 | 0.086 | 21 |
| 5-10 mm | 57 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.263 [0.17, 0.39] | 0.395 | 0.214 | 0.147 | 23 |
| 5-10 mm | 57 | PLAN stage 2 (seg + det + cls), stratified | 0.140 [0.07, 0.25] | 0.381 | 0.144 | 0.072 | 13 |
| 5-10 mm | 57 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.211 [0.12, 0.33] | 0.571 | 0.180 | 0.103 | 9 |
| 10-15 mm | 37 | U-Net, lesion-balanced sampling (stratified split) | 0.568 [0.41, 0.71] | 0.875 | 0.372 | 0.309 | 3 |
| 10-15 mm | 37 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.595 [0.43, 0.74] | 0.710 | 0.444 | 0.366 | 9 |
| 10-15 mm | 37 | PLAN stage 2 (seg + det + cls), stratified | 0.541 [0.38, 0.69] | 0.645 | 0.441 | 0.350 | 11 |
| 10-15 mm | 37 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.595 [0.43, 0.74] | 0.667 | 0.461 | 0.366 | 11 |
| > 15 mm | 106 | U-Net, lesion-balanced sampling (stratified split) | 0.887 [0.81, 0.93] | 0.989 | 0.874 | 0.659 | 1 |
| > 15 mm | 106 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.934 [0.87, 0.97] | 0.980 | 0.876 | 0.692 | 2 |
| > 15 mm | 106 | PLAN stage 2 (seg + det + cls), stratified | 0.915 [0.85, 0.95] | 0.980 | 0.884 | 0.702 | 2 |
| > 15 mm | 106 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.953 [0.89, 0.98] | 0.962 | 0.889 | 0.709 | 4 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 9 |
| **5-15 mm** | 1 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 17 |
| **5-15 mm** | 1 | PLAN stage 2 (seg + det + cls), stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 12 |
| **5-15 mm** | 1 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 15 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls), stratified | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 12 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls), stratified | n/a | 0.000 | n/a | n/a | 7 |
| 5-10 mm | 0 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | n/a | 0.000 | n/a | n/a | 8 |
| 10-15 mm | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 1 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 1 | PLAN stage 2 (seg + det + cls), stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 1 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 7 |
| > 15 mm | 27 | U-Net, lesion-balanced sampling (stratified split) | 1.000 [0.88, 1.00] | 0.844 | 0.869 | 0.776 | 5 |
| > 15 mm | 27 | PLAN stage 1 (seg branch), lesion sampling, stratified | 1.000 [0.88, 1.00] | 0.844 | 0.870 | 0.781 | 5 |
| > 15 mm | 27 | PLAN stage 2 (seg + det + cls), stratified | 1.000 [0.88, 1.00] | 0.871 | 0.861 | 0.773 | 4 |
| > 15 mm | 27 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 1.000 [0.88, 1.00] | 0.844 | 0.862 | 0.774 | 5 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 3 | U-Net, lesion-balanced sampling (stratified split) | 0.333 [0.06, 0.79] | 0.125 | 0.416 | 0.231 | 7 |
| **5-15 mm** | 3 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.667 [0.21, 0.94] | 0.111 | 0.563 | 0.373 | 16 |
| **5-15 mm** | 3 | PLAN stage 2 (seg + det + cls), stratified | 0.333 [0.06, 0.79] | 0.250 | 0.402 | 0.239 | 3 |
| **5-15 mm** | 3 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.333 [0.06, 0.79] | 0.200 | 0.408 | 0.239 | 4 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls), stratified | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 1 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 13 |
| 5-10 mm | 1 | PLAN stage 2 (seg + det + cls), stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 1 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 2 |
| 10-15 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.500 [0.09, 0.91] | 0.167 | 0.458 | 0.347 | 5 |
| 10-15 mm | 2 | PLAN stage 1 (seg branch), lesion sampling, stratified | 1.000 [0.34, 1.00] | 0.400 | 0.619 | 0.560 | 3 |
| 10-15 mm | 2 | PLAN stage 2 (seg + det + cls), stratified | 0.500 [0.09, 0.91] | 0.500 | 0.442 | 0.359 | 1 |
| 10-15 mm | 2 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.500 [0.09, 0.91] | 0.333 | 0.448 | 0.358 | 2 |
| > 15 mm | 21 | U-Net, lesion-balanced sampling (stratified split) | 0.905 [0.71, 0.97] | 0.792 | 0.849 | 0.671 | 5 |
| > 15 mm | 21 | PLAN stage 1 (seg branch), lesion sampling, stratified | 0.905 [0.71, 0.97] | 0.905 | 0.836 | 0.719 | 2 |
| > 15 mm | 21 | PLAN stage 2 (seg + det + cls), stratified | 0.905 [0.71, 0.97] | 0.826 | 0.821 | 0.722 | 4 |
| > 15 mm | 21 | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.905 [0.71, 0.97] | 0.826 | 0.820 | 0.718 | 4 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.260 | 0.332 | 0.394 | 0.375 |
| 5-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.377 | 0.423 | 0.485 |
| 5-15 mm | PLAN stage 2 (seg + det + cls), stratified | 0.132 | 0.243 | 0.314 | 0.356 | 0.307 |
| 5-15 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.186 | 0.288 | 0.367 | 0.444 | 0.353 |
| 5-10 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.121 | 0.182 | 0.234 | 0.231 |
| 5-10 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.249 | 0.283 | 0.373 |
| 5-10 mm | PLAN stage 2 (seg + det + cls), stratified | 0.037 | 0.085 | 0.162 | 0.205 | 0.182 |
| 5-10 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.069 | 0.138 | 0.224 | 0.329 | 0.230 |
| 10-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.463 | 0.550 | 0.625 | 0.584 |
| 10-15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.561 | 0.625 | 0.647 |
| 10-15 mm | PLAN stage 2 (seg + det + cls), stratified | 0.270 | 0.472 | 0.535 | 0.575 | 0.487 |
| 10-15 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.356 | 0.505 | 0.575 | 0.609 | 0.533 |
| > 15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.896 | 0.913 | 0.943 | 0.932 |
| > 15 mm | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.929 | 0.953 | 0.958 |
| > 15 mm | PLAN stage 2 (seg + det + cls), stratified | 0.780 | 0.895 | 0.942 | 0.948 | 0.890 |
| > 15 mm | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.841 | 0.915 | 0.961 | 0.968 | 0.909 |
| all sizes | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.548 | 0.584 | 0.623 | 0.612 |
| all sizes | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.605 | 0.642 | 0.674 |
| all sizes | PLAN stage 2 (seg + det + cls), stratified | 0.438 | 0.537 | 0.586 | 0.609 | 0.563 |
| all sizes | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.485 | 0.565 | 0.618 | 0.658 | 0.591 |
| 0-5 mm (exploratory) | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.066 | 0.094 | 0.113 | 0.123 |
| 0-5 mm (exploratory) | PLAN stage 1 (seg branch), lesion sampling, stratified | n/a | n/a | 0.084 | 0.145 | 0.201 |
| 0-5 mm (exploratory) | PLAN stage 2 (seg + det + cls), stratified | 0.011 | 0.038 | 0.057 | 0.094 | 0.085 |
| 0-5 mm (exploratory) | PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified | 0.004 | 0.061 | 0.085 | 0.158 | 0.107 |

### Training

```
U-Net, lesion-balanced sampling (stratified split): {"epochs_logged": 300, "mean_epoch_seconds": 49.1, "final_val_patch_dice_liver_tumour": [0.9582, 0.8351], "best_val_patch_dice_tumour": 0.8921, "final_train_loss": 0.3521, "synthetic_lesions_per_epoch_mean": 0.0}
PLAN stage 1 (seg branch), lesion sampling, stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.3986, "final_val_loss": -1.3575, "final_global_dice_tumour_liver": [0.8684, 0.9586], "final_online_lesion_sens_prec": [0.6021, 0.4528], "training_hours": 23.4}
PLAN stage 2 (seg + det + cls), stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 5.9154, "final_val_loss": 7.5229, "final_global_dice_tumour_liver": [0.8494, 0.9619], "final_online_lesion_sens_prec": [0.6788, 0.56], "training_hours": 23.0}
PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 5.766, "final_val_loss": 7.5256, "final_global_dice_tumour_liver": [0.855, 0.9587], "final_online_lesion_sens_prec": [0.6236, 0.6687], "training_hours": 22.7}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [193, 214, 246], "current_spacing": [1.0, 1.0, 1.0], "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Artefacts

`results/2026-10-05_plan_stratified_stage2/`: plan1_strat_jobs.json, plan1_strat_plans_summary.json, plan1_strat_test_froc.csv, plan1_strat_test_froc_min33_froc_summary.json, plan1_strat_test_froc_summary.json, plan1_strat_test_per_lesion.csv, plan1_strat_test_size_binned_metrics.csv, plan1_strat_training_log_2026_10_3_11_19_13.txt, plan1_strat_training_log_2026_10_4_10_45_29.txt, plan1_strat_training_log_2026_10_4_10_58_42.txt, plan1_strat_val_froc.csv, plan1_strat_val_froc_min33_froc_summary.json, plan1_strat_val_froc_summary.json, plan1_strat_val_per_lesion.csv, plan1_strat_val_size_binned_metrics.csv, plan2_strat_jobs.json, plan2_strat_plans_summary.json, plan2_strat_test_froc.csv, plan2_strat_test_froc_min33_froc_summary.json, plan2_strat_test_froc_summary.json, plan2_strat_test_per_lesion.csv, plan2_strat_test_size_binned_metrics.csv, plan2_strat_training_log_2026_10_4_11_41_24.txt, plan2_strat_training_log_2026_10_5_10_43_14.txt, plan2_strat_training_log_2026_10_5_11_01_35.txt, plan2_strat_val_froc.csv, plan2_strat_val_froc_min33_froc_summary.json, plan2_strat_val_froc_summary.json, plan2_strat_val_per_lesion.csv, plan2_strat_val_size_binned_metrics.csv, plan2mod_strat_jobs.json, plan2mod_strat_plans_summary.json, plan2mod_strat_test_froc.csv, plan2mod_strat_test_froc_min33_froc_summary.json, plan2mod_strat_test_froc_summary.json, plan2mod_strat_test_per_lesion.csv, plan2mod_strat_test_size_binned_metrics.csv, plan2mod_strat_training_log_2026_10_4_11_41_24.txt, plan2mod_strat_training_log_2026_10_5_10_24_52.txt, plan2mod_strat_training_log_2026_10_5_10_50_11.txt, plan2mod_strat_val_froc.csv, plan2mod_strat_val_froc_min33_froc_summary.json, plan2mod_strat_val_froc_summary.json, plan2mod_strat_val_per_lesion.csv, plan2mod_strat_val_size_binned_metrics.csv, strat_lesion_config.json, strat_lesion_history.json, strat_lesion_jobs.json, strat_lesion_test_evaluation_minsize100_size_binned_metrics.csv, strat_lesion_test_froc.csv, strat_lesion_test_froc_min33_froc_summary.json, strat_lesion_test_froc_summary.json, strat_lesion_test_per_lesion.csv, strat_lesion_test_size_binned_metrics.csv, strat_lesion_val_froc.csv, strat_lesion_val_froc_min33_froc_summary.json, strat_lesion_val_froc_summary.json, strat_lesion_val_per_lesion.csv, strat_lesion_val_size_binned_metrics.csv

