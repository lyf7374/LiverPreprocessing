# 2026-09-29: plan stage2 full model

**Purpose.** Reproduction of PLAN (Yan et al., MICCAI 2023) on unified_v2, stage 2 of 2: the full Pixel-Lesion-pAtient Network (pixel branch + Mask2Former-style lesion branch + patient branch with the det-cls consistency loss), initialised from stage 1 (cfg_hcc_plan1). Compared with the two 300-epoch baselines, the best U-Net modification (CC-DiceCE) and PLAN stage 1.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: small-held-out seed 0 (train 534 / val 67 / test 439); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

Config cfg_hcc_plan2 mirrors cfg_dce_plan2 of PLAN with one lesion class: components (cls, det, seg); FPN pixel decoder with 64-channel mask embeddings; lesion branch = 3-layer transformer decoder with 50 queries over decoder levels 1/16, 1/8, 1/4 (Mask2Former losses: no-object weight 0.1, dice 5, mask 5, cross-entropy 2, 12,544 sampled points, PLAN's foreground-enhanced sampling ratio 2e-4); patient branch = dual-path transformer over encoder levels 1/4 and 1/16 and decoder levels 1/16 and 1/4, 60 patches (patch 112 x 128 x 160 // 32), one patient label (tumour present), started at epoch 50; det-cls consistency loss weight 0.1 from epoch 50. U-Net and FPN weights loaded from stage 1 through seg_pretrain_dir (PLAN's load_from); RAdam 1e-4; PLAN's instance data loader; 300 epochs x 250 iterations (PLAN: continue for 500 more epochs of a 1000-epoch schedule). Inference as in stage 1: the lesion branch's mask predictions replace the pixel branch's softmax (PLAN inference), mirroring, then PLAN's post-processing (largest liver component, lesions farther than 3 mm from the liver removed, minimum lesion volume 25 mm3); raw and post-processed maps evaluated with our evaluator (tumour label 1). Training 21.8 h (4.4 min per epoch), inference + evaluation 1.7 h (job 29076122, 23.5 h).

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `nnunet` = nnU-Net 300 ep; `unet3d` = Plain 3D U-Net 300 ep; `ccdicece` = U-Net + CC-DiceCE (w = 0.5); `plan1` = PLAN stage 1 (seg branch) 300 ep; `plan2` = PLAN stage 2 (seg + det + cls) 300 ep

**Findings.**

The full PLAN is the most precise model so far but not more sensitive than its own stage 1: overall lesion recall 0.440 (stage 1 0.455, CC-DiceCE 0.455, baselines 0.413), lesion precision 0.862 (stage 1 0.797, plain U-Net 0.797), F1 0.583 (stage 1 0.579, CC-DiceCE 0.573, baselines 0.54), false positives per case 0.34 (stage 1 0.55, plain U-Net 0.50); case tumour Dice 0.721 is the highest of all runs, while pooled voxel recall is the lowest (0.777 vs 0.827 for the plain U-Net) because the lesion branch produces tighter masks. By size (stage 1 -> stage 2): 0-5 mm 0.073 -> 0.063, 5-10 mm 0.116 -> 0.101, 10-15 mm 0.512 -> 0.488, > 15 mm 0.906 -> 0.894, with the false-positive components in the 0-5 mm bin down from 42 to 28 and lesion precision in the 5-10 mm bin up from 0.35 to 0.54. Per dataset the precision gain is largest on PLC-CECT (0.651 -> 0.877, 0.15 false positives per case, but recall 0.683 -> 0.610) and WAW-TACE (0.735 -> 0.800 at recall 0.667 -> 0.693). On the validation subset (large lesions only): recall 0.946, precision 0.778 (stage 1 0.607), case Dice 0.717. As in stage 1, PLAN's post-processing only removes false positives (raw: precision 0.680, 0.99 per case, 288 false-positive components in the 0-5 mm bin; post-processed: 0.862, 0.34, 28) and leaves sensitivity unchanged. Conclusion for the small-lesion question: the lesion and patient branches make the detections cleaner but do not recover lesions below 15 mm that the pixel branch misses; none of the PLAN stages exceeds 0.12 sensitivity in the 5-10 mm bin, versus 0.242 with synthetic small tumours in training (experiment 1).

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | U-Net + CC-DiceCE (w = 0.5) | PLAN stage 1 (seg branch) 300 ep | PLAN stage 2 (seg + det + cls) 300 ep |
|---|---:|---|---:|---:|---:|---:|---:|
| overall | 439 | case tumour Dice (mean) | 0.713 | 0.709 | 0.703 | 0.716 | 0.721 |
|  |  | voxel Dice (pooled) | 0.828 | 0.851 | 0.847 | 0.834 | 0.825 |
|  |  | voxel recall (pooled) | 0.795 | 0.827 | 0.833 | 0.801 | 0.777 |
|  |  | voxel precision (pooled) | 0.865 | 0.877 | 0.861 | 0.870 | 0.880 |
|  |  | lesion recall = sensitivity | 0.413 | 0.413 | 0.455 | 0.455 | 0.440 |
|  |  | lesion precision | 0.786 | 0.797 | 0.774 | 0.797 | 0.862 |
|  |  | lesion F1 | 0.542 | 0.544 | 0.573 | 0.579 | 0.583 |
|  |  | false positives per case | 0.54 | 0.50 | 0.63 | 0.55 | 0.34 |
| MCT-LTDiag | 354 | case tumour Dice (mean) | 0.733 | 0.730 | 0.725 | 0.734 | 0.737 |
|  |  | voxel Dice (pooled) | 0.835 | 0.856 | 0.852 | 0.838 | 0.833 |
|  |  | voxel recall (pooled) | 0.799 | 0.827 | 0.831 | 0.798 | 0.778 |
|  |  | voxel precision (pooled) | 0.874 | 0.887 | 0.874 | 0.882 | 0.895 |
|  |  | lesion recall = sensitivity | 0.398 | 0.396 | 0.436 | 0.437 | 0.423 |
|  |  | lesion precision | 0.799 | 0.821 | 0.800 | 0.813 | 0.865 |
|  |  | lesion F1 | 0.531 | 0.534 | 0.565 | 0.568 | 0.569 |
|  |  | false positives per case | 0.55 | 0.47 | 0.60 | 0.55 | 0.36 |
| PLC-CECT | 48 | case tumour Dice (mean) | 0.540 | 0.537 | 0.517 | 0.567 | 0.571 |
|  |  | voxel Dice (pooled) | 0.801 | 0.843 | 0.839 | 0.832 | 0.802 |
|  |  | voxel recall (pooled) | 0.763 | 0.835 | 0.850 | 0.802 | 0.752 |
|  |  | voxel precision (pooled) | 0.843 | 0.852 | 0.829 | 0.866 | 0.860 |
|  |  | lesion recall = sensitivity | 0.634 | 0.659 | 0.744 | 0.683 | 0.610 |
|  |  | lesion precision | 0.703 | 0.635 | 0.616 | 0.651 | 0.877 |
|  |  | lesion F1 | 0.667 | 0.647 | 0.674 | 0.667 | 0.719 |
|  |  | false positives per case | 0.46 | 0.65 | 0.79 | 0.62 | 0.15 |
| WAW-TACE | 37 | case tumour Dice (mean) | 0.727 | 0.717 | 0.713 | 0.718 | 0.725 |
|  |  | voxel Dice (pooled) | 0.801 | 0.800 | 0.797 | 0.783 | 0.775 |
|  |  | voxel recall (pooled) | 0.807 | 0.810 | 0.826 | 0.837 | 0.818 |
|  |  | voxel precision (pooled) | 0.795 | 0.790 | 0.769 | 0.736 | 0.736 |
|  |  | lesion recall = sensitivity | 0.573 | 0.587 | 0.613 | 0.667 | 0.693 |
|  |  | lesion precision | 0.683 | 0.667 | 0.613 | 0.735 | 0.800 |
|  |  | lesion F1 | 0.623 | 0.624 | 0.613 | 0.699 | 0.743 |
|  |  | false positives per case | 0.54 | 0.59 | 0.78 | 0.49 | 0.35 |

### Per size bin

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 509 | nnU-Net 300 ep | 0.051 [0.04, 0.07] | 0.252 | 0.048 | 0.000 | 77 |
| 0-5 mm | 509 | Plain 3D U-Net 300 ep | 0.065 [0.05, 0.09] | 0.337 | 0.051 | 0.001 | 65 |
| 0-5 mm | 509 | U-Net + CC-DiceCE (w = 0.5) | 0.083 [0.06, 0.11] | 0.365 | 0.062 | 0.001 | 73 |
| 0-5 mm | 509 | PLAN stage 1 (seg branch) 300 ep | 0.073 [0.05, 0.10] | 0.468 | 0.060 | 0.001 | 42 |
| 0-5 mm | 509 | PLAN stage 2 (seg + det + cls) 300 ep | 0.063 [0.04, 0.09] | 0.533 | 0.055 | 0.001 | 28 |
| 5-10 mm | 475 | nnU-Net 300 ep | 0.078 [0.06, 0.11] | 0.366 | 0.066 | 0.027 | 64 |
| 5-10 mm | 475 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.312 | 0.058 | 0.019 | 64 |
| 5-10 mm | 475 | U-Net + CC-DiceCE (w = 0.5) | 0.105 [0.08, 0.14] | 0.373 | 0.097 | 0.040 | 84 |
| 5-10 mm | 475 | PLAN stage 1 (seg branch) 300 ep | 0.116 [0.09, 0.15] | 0.348 | 0.109 | 0.045 | 103 |
| 5-10 mm | 475 | PLAN stage 2 (seg + det + cls) 300 ep | 0.101 [0.08, 0.13] | 0.539 | 0.103 | 0.038 | 41 |
| 10-15 mm | 365 | nnU-Net 300 ep | 0.425 [0.37, 0.48] | 0.791 | 0.314 | 0.237 | 41 |
| 10-15 mm | 365 | Plain 3D U-Net 300 ep | 0.441 [0.39, 0.49] | 0.785 | 0.322 | 0.242 | 44 |
| 10-15 mm | 365 | U-Net + CC-DiceCE (w = 0.5) | 0.529 [0.48, 0.58] | 0.739 | 0.404 | 0.292 | 68 |
| 10-15 mm | 365 | PLAN stage 1 (seg branch) 300 ep | 0.512 [0.46, 0.56] | 0.760 | 0.371 | 0.280 | 59 |
| 10-15 mm | 365 | PLAN stage 2 (seg + det + cls) 300 ep | 0.488 [0.44, 0.54] | 0.805 | 0.355 | 0.276 | 43 |
| > 15 mm | 742 | nnU-Net 300 ep | 0.871 [0.84, 0.89] | 0.924 | 0.797 | 0.646 | 53 |
| > 15 mm | 742 | Plain 3D U-Net 300 ep | 0.863 [0.84, 0.89] | 0.932 | 0.830 | 0.640 | 47 |
| > 15 mm | 742 | U-Net + CC-DiceCE (w = 0.5) | 0.898 [0.87, 0.92] | 0.926 | 0.836 | 0.660 | 53 |
| > 15 mm | 742 | PLAN stage 1 (seg branch) 300 ep | 0.906 [0.88, 0.92] | 0.946 | 0.803 | 0.668 | 38 |
| > 15 mm | 742 | PLAN stage 2 (seg + det + cls) 300 ep | 0.894 [0.87, 0.91] | 0.948 | 0.779 | 0.677 | 36 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 508 | nnU-Net 300 ep | 0.049 [0.03, 0.07] | 0.309 | 0.048 | 0.000 | 56 |
| 0-5 mm | 508 | Plain 3D U-Net 300 ep | 0.063 [0.04, 0.09] | 0.421 | 0.051 | 0.001 | 44 |
| 0-5 mm | 508 | U-Net + CC-DiceCE (w = 0.5) | 0.081 [0.06, 0.11] | 0.461 | 0.062 | 0.001 | 48 |
| 0-5 mm | 508 | PLAN stage 1 (seg branch) 300 ep | 0.071 [0.05, 0.10] | 0.507 | 0.060 | 0.001 | 35 |
| 0-5 mm | 508 | PLAN stage 2 (seg + det + cls) 300 ep | 0.061 [0.04, 0.09] | 0.564 | 0.054 | 0.001 | 24 |
| 5-10 mm | 458 | nnU-Net 300 ep | 0.079 [0.06, 0.11] | 0.375 | 0.068 | 0.028 | 60 |
| 5-10 mm | 458 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.346 | 0.059 | 0.020 | 53 |
| 5-10 mm | 458 | U-Net + CC-DiceCE (w = 0.5) | 0.100 [0.08, 0.13] | 0.407 | 0.095 | 0.039 | 67 |
| 5-10 mm | 458 | PLAN stage 1 (seg branch) 300 ep | 0.116 [0.09, 0.15] | 0.384 | 0.110 | 0.045 | 85 |
| 5-10 mm | 458 | PLAN stage 2 (seg + det + cls) 300 ep | 0.100 [0.08, 0.13] | 0.535 | 0.101 | 0.039 | 40 |
| 10-15 mm | 323 | nnU-Net 300 ep | 0.440 [0.39, 0.49] | 0.798 | 0.318 | 0.244 | 36 |
| 10-15 mm | 323 | Plain 3D U-Net 300 ep | 0.455 [0.40, 0.51] | 0.817 | 0.330 | 0.249 | 33 |
| 10-15 mm | 323 | U-Net + CC-DiceCE (w = 0.5) | 0.542 [0.49, 0.60] | 0.754 | 0.410 | 0.298 | 57 |
| 10-15 mm | 323 | PLAN stage 1 (seg branch) 300 ep | 0.526 [0.47, 0.58] | 0.776 | 0.379 | 0.288 | 49 |
| 10-15 mm | 323 | PLAN stage 2 (seg + det + cls) 300 ep | 0.505 [0.45, 0.56] | 0.819 | 0.369 | 0.290 | 36 |
| > 15 mm | 645 | nnU-Net 300 ep | 0.878 [0.85, 0.90] | 0.932 | 0.802 | 0.651 | 41 |
| > 15 mm | 645 | Plain 3D U-Net 300 ep | 0.865 [0.84, 0.89] | 0.938 | 0.830 | 0.643 | 37 |
| > 15 mm | 645 | U-Net + CC-DiceCE (w = 0.5) | 0.902 [0.88, 0.92] | 0.937 | 0.834 | 0.665 | 39 |
| > 15 mm | 645 | PLAN stage 1 (seg branch) 300 ep | 0.909 [0.88, 0.93] | 0.959 | 0.801 | 0.672 | 25 |
| > 15 mm | 645 | PLAN stage 2 (seg + det + cls) 300 ep | 0.898 [0.87, 0.92] | 0.954 | 0.781 | 0.682 | 28 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 1 | nnU-Net 300 ep | 1.000 [0.21, 1.00] | 0.091 | 1.000 | 0.000 | 10 |
| 0-5 mm | 1 | Plain 3D U-Net 300 ep | 1.000 [0.21, 1.00] | 0.067 | 1.000 | 0.000 | 14 |
| 0-5 mm | 1 | U-Net + CC-DiceCE (w = 0.5) | 1.000 [0.21, 1.00] | 0.077 | 1.000 | 0.000 | 12 |
| 0-5 mm | 1 | PLAN stage 1 (seg branch) 300 ep | 1.000 [0.21, 1.00] | 0.250 | 1.000 | 0.000 | 3 |
| 0-5 mm | 1 | PLAN stage 2 (seg + det + cls) 300 ep | 1.000 [0.21, 1.00] | 0.500 | 1.000 | 0.000 | 1 |
| 5-10 mm | 9 | nnU-Net 300 ep | 0.111 [0.02, 0.44] | 0.333 | 0.061 | 0.000 | 2 |
| 5-10 mm | 9 | Plain 3D U-Net 300 ep | 0.111 [0.02, 0.44] | 0.143 | 0.087 | 0.000 | 6 |
| 5-10 mm | 9 | U-Net + CC-DiceCE (w = 0.5) | 0.444 [0.19, 0.73] | 0.250 | 0.222 | 0.113 | 12 |
| 5-10 mm | 9 | PLAN stage 1 (seg branch) 300 ep | 0.111 [0.02, 0.44] | 0.067 | 0.117 | 0.000 | 14 |
| 5-10 mm | 9 | PLAN stage 2 (seg + det + cls) 300 ep | 0.111 [0.02, 0.44] | 1.000 | 0.131 | 0.000 | 0 |
| 10-15 mm | 23 | nnU-Net 300 ep | 0.391 [0.22, 0.59] | 0.750 | 0.333 | 0.229 | 3 |
| 10-15 mm | 23 | Plain 3D U-Net 300 ep | 0.435 [0.26, 0.63] | 0.588 | 0.306 | 0.232 | 7 |
| 10-15 mm | 23 | U-Net + CC-DiceCE (w = 0.5) | 0.609 [0.41, 0.78] | 0.609 | 0.486 | 0.330 | 9 |
| 10-15 mm | 23 | PLAN stage 1 (seg branch) 300 ep | 0.478 [0.29, 0.67] | 0.611 | 0.384 | 0.261 | 7 |
| 10-15 mm | 23 | PLAN stage 2 (seg + det + cls) 300 ep | 0.391 [0.22, 0.59] | 0.750 | 0.251 | 0.170 | 3 |
| > 15 mm | 49 | nnU-Net 300 ep | 0.837 [0.71, 0.91] | 0.854 | 0.764 | 0.564 | 7 |
| > 15 mm | 49 | Plain 3D U-Net 300 ep | 0.857 [0.73, 0.93] | 0.913 | 0.836 | 0.566 | 4 |
| > 15 mm | 49 | U-Net + CC-DiceCE (w = 0.5) | 0.857 [0.73, 0.93] | 0.894 | 0.851 | 0.561 | 5 |
| > 15 mm | 49 | PLAN stage 1 (seg branch) 300 ep | 0.878 [0.76, 0.94] | 0.878 | 0.803 | 0.600 | 6 |
| > 15 mm | 49 | PLAN stage 2 (seg + det + cls) 300 ep | 0.796 [0.66, 0.89] | 0.929 | 0.753 | 0.559 | 3 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 0-5 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 8 | nnU-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 8 | Plain 3D U-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 8 | U-Net + CC-DiceCE (w = 0.5) | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 8 | PLAN stage 1 (seg branch) 300 ep | 0.125 [0.02, 0.47] | 0.200 | 0.082 | 0.073 | 4 |
| 5-10 mm | 8 | PLAN stage 2 (seg + det + cls) 300 ep | 0.125 [0.02, 0.47] | 0.500 | 0.146 | 0.052 | 1 |
| 10-15 mm | 19 | nnU-Net 300 ep | 0.211 [0.09, 0.43] | 0.667 | 0.227 | 0.141 | 2 |
| 10-15 mm | 19 | Plain 3D U-Net 300 ep | 0.211 [0.09, 0.43] | 0.500 | 0.219 | 0.134 | 4 |
| 10-15 mm | 19 | U-Net + CC-DiceCE (w = 0.5) | 0.211 [0.09, 0.43] | 0.667 | 0.218 | 0.131 | 2 |
| 10-15 mm | 19 | PLAN stage 1 (seg branch) 300 ep | 0.316 [0.15, 0.54] | 0.667 | 0.230 | 0.167 | 3 |
| 10-15 mm | 19 | PLAN stage 2 (seg + det + cls) 300 ep | 0.316 [0.15, 0.54] | 0.600 | 0.251 | 0.169 | 4 |
| > 15 mm | 48 | nnU-Net 300 ep | 0.812 [0.68, 0.90] | 0.886 | 0.810 | 0.655 | 5 |
| > 15 mm | 48 | Plain 3D U-Net 300 ep | 0.833 [0.70, 0.91] | 0.870 | 0.813 | 0.675 | 6 |
| > 15 mm | 48 | U-Net + CC-DiceCE (w = 0.5) | 0.875 [0.75, 0.94] | 0.824 | 0.829 | 0.688 | 9 |
| > 15 mm | 48 | PLAN stage 1 (seg branch) 300 ep | 0.896 [0.78, 0.95] | 0.860 | 0.840 | 0.688 | 7 |
| > 15 mm | 48 | PLAN stage 2 (seg + det + cls) 300 ep | 0.938 [0.83, 0.98] | 0.900 | 0.820 | 0.730 | 5 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
PLAN stage 1 (seg branch) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.4165, "final_val_loss": -1.2835, "final_global_dice_tumour_liver": [0.8185, 0.9638], "final_online_lesion_sens_prec": [0.871, 0.4426], "training_hours": 21.7}
PLAN stage 2 (seg + det + cls) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 4.3773, "final_val_loss": 5.4606, "final_global_dice_tumour_liver": [0.8544, 0.9649], "final_online_lesion_sens_prec": [0.8256, 0.6174], "training_hours": 21.8}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [183, 218, 246], "current_spacing": [1.0, 1.0, 1.0], "do_dummy_2D_data_aug": false, "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | U-Net + CC-DiceCE (w = 0.5) | PLAN stage 1 (seg branch) 300 ep | PLAN stage 2 (seg + det + cls) 300 ep |
|---|---:|---|---:|---:|---:|---:|---:|
| overall | 67 | case tumour Dice (mean) | 0.713 | 0.702 | 0.670 | 0.696 | 0.717 |
|  |  | voxel Dice (pooled) | 0.807 | 0.784 | 0.782 | 0.806 | 0.813 |
|  |  | voxel recall (pooled) | 0.757 | 0.753 | 0.764 | 0.773 | 0.764 |
|  |  | voxel precision (pooled) | 0.865 | 0.818 | 0.802 | 0.842 | 0.869 |
|  |  | lesion recall = sensitivity | 0.973 | 0.959 | 0.932 | 0.959 | 0.946 |
|  |  | lesion precision | 0.626 | 0.617 | 0.473 | 0.607 | 0.778 |
|  |  | lesion F1 | 0.762 | 0.751 | 0.627 | 0.743 | 0.854 |
|  |  | false positives per case | 0.64 | 0.66 | 1.15 | 0.69 | 0.30 |
| MCT-LTDiag | 19 | case tumour Dice (mean) | 0.788 | 0.788 | 0.787 | 0.798 | 0.795 |
|  |  | voxel Dice (pooled) | 0.863 | 0.859 | 0.866 | 0.871 | 0.877 |
|  |  | voxel recall (pooled) | 0.854 | 0.851 | 0.883 | 0.871 | 0.867 |
|  |  | voxel precision (pooled) | 0.872 | 0.868 | 0.850 | 0.871 | 0.888 |
|  |  | lesion recall = sensitivity | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 |
|  |  | lesion precision | 0.750 | 0.774 | 0.706 | 0.706 | 0.800 |
|  |  | lesion F1 | 0.842 | 0.857 | 0.814 | 0.814 | 0.873 |
|  |  | false positives per case | 0.42 | 0.37 | 0.53 | 0.53 | 0.32 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.642 | 0.614 | 0.571 | 0.601 | 0.643 |
|  |  | voxel Dice (pooled) | 0.797 | 0.751 | 0.744 | 0.780 | 0.789 |
|  |  | voxel recall (pooled) | 0.742 | 0.734 | 0.738 | 0.752 | 0.739 |
|  |  | voxel precision (pooled) | 0.861 | 0.770 | 0.751 | 0.811 | 0.846 |
|  |  | lesion recall = sensitivity | 0.963 | 0.926 | 0.889 | 0.926 | 0.889 |
|  |  | lesion precision | 0.553 | 0.446 | 0.282 | 0.532 | 0.727 |
|  |  | lesion F1 | 0.703 | 0.602 | 0.429 | 0.676 | 0.800 |
|  |  | false positives per case | 0.62 | 0.91 | 1.79 | 0.65 | 0.26 |
| WAW-TACE | 14 | case tumour Dice (mean) | 0.747 | 0.757 | 0.707 | 0.748 | 0.745 |
|  |  | voxel Dice (pooled) | 0.788 | 0.817 | 0.822 | 0.825 | 0.827 |
|  |  | voxel recall (pooled) | 0.721 | 0.729 | 0.742 | 0.755 | 0.753 |
|  |  | voxel precision (pooled) | 0.868 | 0.929 | 0.922 | 0.910 | 0.918 |
|  |  | lesion recall = sensitivity | 1.000 | 1.000 | 0.955 | 1.000 | 1.000 |
|  |  | lesion precision | 0.611 | 0.786 | 0.778 | 0.611 | 0.815 |
|  |  | lesion F1 | 0.759 | 0.880 | 0.857 | 0.759 | 0.898 |
|  |  | false positives per case | 1.00 | 0.43 | 0.43 | 1.00 | 0.36 |

### Per size bin

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 18 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 26 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| 0-5 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 10 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 18 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 21 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| > 15 mm | 74 | nnU-Net 300 ep | 0.973 [0.91, 0.99] | 0.889 | 0.757 | 0.727 | 9 |
| > 15 mm | 74 | Plain 3D U-Net 300 ep | 0.959 [0.89, 0.99] | 0.910 | 0.753 | 0.726 | 7 |
| > 15 mm | 74 | U-Net + CC-DiceCE (w = 0.5) | 0.932 [0.85, 0.97] | 0.852 | 0.764 | 0.708 | 12 |
| > 15 mm | 74 | PLAN stage 1 (seg branch) 300 ep | 0.959 [0.89, 0.99] | 0.835 | 0.773 | 0.727 | 14 |
| > 15 mm | 74 | PLAN stage 2 (seg + det + cls) 300 ep | 0.946 [0.87, 0.98] | 0.897 | 0.764 | 0.718 | 8 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | n/a | n/a | n/a | 0 |
| 0-5 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | n/a | n/a | n/a | 0 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 8 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| > 15 mm | 25 | nnU-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.854 | 0.772 | 0 |
| > 15 mm | 25 | Plain 3D U-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.851 | 0.776 | 0 |
| > 15 mm | 25 | U-Net + CC-DiceCE (w = 0.5) | 0.960 [0.80, 0.99] | 0.923 | 0.883 | 0.783 | 2 |
| > 15 mm | 25 | PLAN stage 1 (seg branch) 300 ep | 0.960 [0.80, 0.99] | 0.960 | 0.871 | 0.784 | 1 |
| > 15 mm | 25 | PLAN stage 2 (seg + det + cls) 300 ep | 0.960 [0.80, 0.99] | 0.960 | 0.867 | 0.784 | 1 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 24 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 0-5 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 17 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 11 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | n/a | n/a | n/a | 0 |
| > 15 mm | 27 | nnU-Net 300 ep | 0.963 [0.82, 0.99] | 0.788 | 0.742 | 0.694 | 7 |
| > 15 mm | 27 | Plain 3D U-Net 300 ep | 0.926 [0.77, 0.98] | 0.781 | 0.734 | 0.681 | 7 |
| > 15 mm | 27 | U-Net + CC-DiceCE (w = 0.5) | 0.889 [0.72, 0.96] | 0.727 | 0.738 | 0.648 | 9 |
| > 15 mm | 27 | PLAN stage 1 (seg branch) 300 ep | 0.926 [0.77, 0.98] | 0.714 | 0.752 | 0.677 | 10 |
| > 15 mm | 27 | PLAN stage 2 (seg + det + cls) 300 ep | 0.889 [0.72, 0.96] | 0.800 | 0.739 | 0.658 | 6 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | n/a | n/a | n/a | 0 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| > 15 mm | 22 | nnU-Net 300 ep | 1.000 [0.85, 1.00] | 0.917 | 0.721 | 0.716 | 2 |
| > 15 mm | 22 | Plain 3D U-Net 300 ep | 1.000 [0.85, 1.00] | 1.000 | 0.729 | 0.724 | 0 |
| > 15 mm | 22 | U-Net + CC-DiceCE (w = 0.5) | 0.955 [0.78, 0.99] | 0.955 | 0.742 | 0.697 | 1 |
| > 15 mm | 22 | PLAN stage 1 (seg branch) 300 ep | 1.000 [0.85, 1.00] | 0.880 | 0.755 | 0.722 | 3 |
| > 15 mm | 22 | PLAN stage 2 (seg + det + cls) 300 ep | 1.000 [0.85, 1.00] | 0.957 | 0.753 | 0.716 | 1 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
PLAN stage 1 (seg branch) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.4165, "final_val_loss": -1.2835, "final_global_dice_tumour_liver": [0.8185, 0.9638], "final_online_lesion_sens_prec": [0.871, 0.4426], "training_hours": 21.7}
PLAN stage 2 (seg + det + cls) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 4.3773, "final_val_loss": 5.4606, "final_global_dice_tumour_liver": [0.8544, 0.9649], "final_online_lesion_sens_prec": [0.8256, 0.6174], "training_hours": 21.8}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [183, 218, 246], "current_spacing": [1.0, 1.0, 1.0], "do_dummy_2D_data_aug": false, "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Artefacts

`results/2026-09-29_plan_stage2_full_model/`: ccdicece_config.json, ccdicece_history.json, ccdicece_jobs.json, ccdicece_test_per_lesion.csv, ccdicece_test_size_binned_metrics.csv, ccdicece_val_per_lesion.csv, ccdicece_val_size_binned_metrics.csv, nnunet_cohorts.json, nnunet_nnUNetPlans.json, nnunet_test_per_lesion.csv, nnunet_test_size_binned_metrics.csv, nnunet_training_log_2026_9_27_02_50_22.txt, nnunet_val_per_lesion.csv, nnunet_val_size_binned_metrics.csv, plan1_jobs.json, plan1_plans_summary.json, plan1_test_per_lesion.csv, plan1_test_size_binned_metrics.csv, plan1_training_log_2026_9_27_22_13_27.txt, plan1_training_log_2026_9_28_19_52_43.txt, plan1_training_log_2026_9_28_20_00_48.txt, plan1_val_per_lesion.csv, plan1_val_size_binned_metrics.csv, plan2_jobs.json, plan2_plans_summary.json, plan2_test_per_lesion.csv, plan2_test_size_binned_metrics.csv, plan2_training_log_2026_9_28_21_11_28.txt, plan2_training_log_2026_9_29_18_59_52.txt, plan2_training_log_2026_9_29_19_10_58.txt, plan2_val_per_lesion.csv, plan2_val_size_binned_metrics.csv, unet3d_config.json, unet3d_history.json, unet3d_test_per_lesion.csv, unet3d_test_size_binned_metrics.csv, unet3d_val_per_lesion.csv, unet3d_val_size_binned_metrics.csv

