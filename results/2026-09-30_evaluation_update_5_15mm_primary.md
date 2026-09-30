# 2026-09-30: evaluation update 5 15mm primary

**Purpose.** Re-read all seven runs on the small-held-out split with a revised evaluation focus: the 5-15 mm bin becomes the primary small-lesion target and the 0-5 mm bin is reported as exploratory only, following an audit of the ground truth. No new training; the tables below are recomputed from the existing predictions.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: small-held-out seed 0 (train 534 / val 67 / test 439); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

Ground-truth audit of the test subset (per_lesion.csv of the plain U-Net evaluation): the 0-5 mm bin holds 509 components, of which 236 (46 %) are <= 10 voxels (equivalent spherical diameter <= 2.7 mm) and 141 are <= 5 voxels; all 509 lie in cases that also contain a lesion >= 10 mm (460 with a lesion > 15 mm), 508 of them in MCT-LTDiag; no test case has only 0-5 mm lesions. These components are mostly annotation fragments or detached parts of larger lesions rather than independent small tumours. The 5-10 mm bin (475 lesions) is also almost entirely MCT-LTDiag (458; PLC-CECT 9, WAW-TACE 8) and 414 of them lie in cases with a lesion > 15 mm, so the small-lesion benchmark is effectively a multi-lesion liver-metastasis benchmark.

Revised reporting (tools/analyze_results.py): a derived 5-15 mm row (union of 5-10 and 10-15 mm: 840 lesions on the test subset) is the primary small-lesion metric; 0-5 mm rows are labelled exploratory; a FROC table (sensitivity at 0.25 / 0.5 / 1 / 2 false positives per case from saved probability maps, scripts/froc_by_lesion_size.py) is added for runs that have probability maps. The FROC curves and a ceiling experiment (plain U-Net on the stratified split, which contains small lesions in training) are running and will be recorded separately.

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `nnunet` = nnU-Net 300 ep; `unet3d` = Plain 3D U-Net 300 ep; `synth` = U-Net + synthetic small tumours; `ccdicece` = U-Net + CC-DiceCE (w = 0.5); `disentangled` = U-Net + phase disentanglement (DisC-Diff style); `plan1` = PLAN stage 1 (seg branch) 300 ep; `plan2` = PLAN stage 2 (seg + det + cls) 300 ep

**Findings.**

Primary metric, 5-15 mm sensitivity [95 % CI] / lesion precision / false-positive components in the bin (test subset, 840 lesions): nnU-Net 0.229 [0.20, 0.26] / 0.646 / 105; plain 3D U-Net 0.226 / 0.638 / 108; + synthetic small tumours 0.405 [0.37, 0.44] / 0.473 / 379; + CC-DiceCE 0.289 / 0.615 / 152; + phase disentanglement 0.206 / 0.625 / 104; PLAN stage 1 0.288 / 0.599 / 162; PLAN stage 2 0.269 / 0.729 / 84. With this focus the ranking is unchanged but the margins are clearer: synthetic training almost doubles the 5-15 mm sensitivity of the baselines at the cost of 3.5x the false-positive components in that size range; CC-DiceCE and PLAN stage 1 gain about 6 points at similar precision; PLAN stage 2 trades 2 points of sensitivity for the best precision. Every model trained only on lesions > 15 mm stays below 0.30 sensitivity in the 5-15 mm range except the synthetic run, which is the motivation for the ceiling experiment on the stratified split.

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | U-Net + synthetic small tumours | U-Net + CC-DiceCE (w = 0.5) | U-Net + phase disentanglement (DisC-Diff style) | PLAN stage 1 (seg branch) 300 ep | PLAN stage 2 (seg + det + cls) 300 ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 439 | case tumour Dice (mean) | 0.713 | 0.709 | 0.704 | 0.703 | 0.709 | 0.716 | 0.721 |
|  |  | voxel Dice (pooled) | 0.828 | 0.851 | 0.847 | 0.847 | 0.841 | 0.834 | 0.825 |
|  |  | voxel recall (pooled) | 0.795 | 0.827 | 0.832 | 0.833 | 0.817 | 0.801 | 0.777 |
|  |  | voxel precision (pooled) | 0.865 | 0.877 | 0.863 | 0.861 | 0.868 | 0.870 | 0.880 |
|  |  | lesion recall = sensitivity | 0.413 | 0.413 | 0.508 | 0.455 | 0.406 | 0.455 | 0.440 |
|  |  | lesion precision | 0.786 | 0.797 | 0.583 | 0.774 | 0.787 | 0.797 | 0.862 |
|  |  | lesion F1 | 0.542 | 0.544 | 0.543 | 0.573 | 0.536 | 0.579 | 0.583 |
|  |  | false positives per case | 0.54 | 0.50 | 1.73 | 0.63 | 0.52 | 0.55 | 0.34 |
| MCT-LTDiag | 354 | case tumour Dice (mean) | 0.733 | 0.730 | 0.726 | 0.725 | 0.730 | 0.734 | 0.737 |
|  |  | voxel Dice (pooled) | 0.835 | 0.856 | 0.851 | 0.852 | 0.846 | 0.838 | 0.833 |
|  |  | voxel recall (pooled) | 0.799 | 0.827 | 0.830 | 0.831 | 0.813 | 0.798 | 0.778 |
|  |  | voxel precision (pooled) | 0.874 | 0.887 | 0.873 | 0.874 | 0.882 | 0.882 | 0.895 |
|  |  | lesion recall = sensitivity | 0.398 | 0.396 | 0.488 | 0.436 | 0.388 | 0.437 | 0.423 |
|  |  | lesion precision | 0.799 | 0.821 | 0.628 | 0.800 | 0.807 | 0.813 | 0.865 |
|  |  | lesion F1 | 0.531 | 0.534 | 0.549 | 0.565 | 0.524 | 0.568 | 0.569 |
|  |  | false positives per case | 0.55 | 0.47 | 1.58 | 0.60 | 0.51 | 0.55 | 0.36 |
| PLC-CECT | 48 | case tumour Dice (mean) | 0.540 | 0.537 | 0.516 | 0.517 | 0.529 | 0.567 | 0.571 |
|  |  | voxel Dice (pooled) | 0.801 | 0.843 | 0.843 | 0.839 | 0.841 | 0.832 | 0.802 |
|  |  | voxel recall (pooled) | 0.763 | 0.835 | 0.844 | 0.850 | 0.840 | 0.802 | 0.752 |
|  |  | voxel precision (pooled) | 0.843 | 0.852 | 0.842 | 0.829 | 0.843 | 0.866 | 0.860 |
|  |  | lesion recall = sensitivity | 0.634 | 0.659 | 0.805 | 0.744 | 0.671 | 0.683 | 0.610 |
|  |  | lesion precision | 0.703 | 0.635 | 0.307 | 0.616 | 0.632 | 0.651 | 0.877 |
|  |  | lesion F1 | 0.667 | 0.647 | 0.444 | 0.674 | 0.651 | 0.667 | 0.719 |
|  |  | false positives per case | 0.46 | 0.65 | 3.10 | 0.79 | 0.67 | 0.62 | 0.15 |
| WAW-TACE | 37 | case tumour Dice (mean) | 0.727 | 0.717 | 0.725 | 0.713 | 0.722 | 0.718 | 0.725 |
|  |  | voxel Dice (pooled) | 0.801 | 0.800 | 0.803 | 0.797 | 0.774 | 0.783 | 0.775 |
|  |  | voxel recall (pooled) | 0.807 | 0.810 | 0.828 | 0.826 | 0.808 | 0.837 | 0.818 |
|  |  | voxel precision (pooled) | 0.795 | 0.790 | 0.779 | 0.769 | 0.744 | 0.736 | 0.736 |
|  |  | lesion recall = sensitivity | 0.573 | 0.587 | 0.693 | 0.613 | 0.587 | 0.667 | 0.693 |
|  |  | lesion precision | 0.683 | 0.667 | 0.505 | 0.613 | 0.698 | 0.735 | 0.800 |
|  |  | lesion F1 | 0.623 | 0.624 | 0.584 | 0.613 | 0.638 | 0.699 | 0.743 |
|  |  | false positives per case | 0.54 | 0.59 | 1.38 | 0.78 | 0.51 | 0.49 | 0.35 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 840 | nnU-Net 300 ep | 0.229 [0.20, 0.26] | 0.646 | 0.259 | 0.118 | 105 |
| **5-15 mm** | 840 | Plain 3D U-Net 300 ep | 0.226 [0.20, 0.26] | 0.638 | 0.264 | 0.116 | 108 |
| **5-15 mm** | 840 | U-Net + synthetic small tumours | 0.405 [0.37, 0.44] | 0.473 | 0.361 | 0.223 | 379 |
| **5-15 mm** | 840 | U-Net + CC-DiceCE (w = 0.5) | 0.289 [0.26, 0.32] | 0.615 | 0.336 | 0.149 | 152 |
| **5-15 mm** | 840 | U-Net + phase disentanglement (DisC-Diff style) | 0.206 [0.18, 0.23] | 0.625 | 0.251 | 0.108 | 104 |
| **5-15 mm** | 840 | PLAN stage 1 (seg branch) 300 ep | 0.288 [0.26, 0.32] | 0.599 | 0.313 | 0.147 | 162 |
| **5-15 mm** | 840 | PLAN stage 2 (seg + det + cls) 300 ep | 0.269 [0.24, 0.30] | 0.729 | 0.300 | 0.142 | 84 |
| 0-5 mm (exploratory) | 509 | nnU-Net 300 ep | 0.051 [0.04, 0.07] | 0.252 | 0.048 | 0.000 | 77 |
| 0-5 mm (exploratory) | 509 | Plain 3D U-Net 300 ep | 0.065 [0.05, 0.09] | 0.337 | 0.051 | 0.001 | 65 |
| 0-5 mm (exploratory) | 509 | U-Net + synthetic small tumours | 0.124 [0.10, 0.16] | 0.157 | 0.076 | 0.011 | 338 |
| 0-5 mm (exploratory) | 509 | U-Net + CC-DiceCE (w = 0.5) | 0.083 [0.06, 0.11] | 0.365 | 0.062 | 0.001 | 73 |
| 0-5 mm (exploratory) | 509 | U-Net + phase disentanglement (DisC-Diff style) | 0.061 [0.04, 0.09] | 0.272 | 0.038 | 0.000 | 83 |
| 0-5 mm (exploratory) | 509 | PLAN stage 1 (seg branch) 300 ep | 0.073 [0.05, 0.10] | 0.468 | 0.060 | 0.001 | 42 |
| 0-5 mm (exploratory) | 509 | PLAN stage 2 (seg + det + cls) 300 ep | 0.063 [0.04, 0.09] | 0.533 | 0.055 | 0.001 | 28 |
| 5-10 mm | 475 | nnU-Net 300 ep | 0.078 [0.06, 0.11] | 0.366 | 0.066 | 0.027 | 64 |
| 5-10 mm | 475 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.312 | 0.058 | 0.019 | 64 |
| 5-10 mm | 475 | U-Net + synthetic small tumours | 0.242 [0.21, 0.28] | 0.288 | 0.171 | 0.116 | 284 |
| 5-10 mm | 475 | U-Net + CC-DiceCE (w = 0.5) | 0.105 [0.08, 0.14] | 0.373 | 0.097 | 0.040 | 84 |
| 5-10 mm | 475 | U-Net + phase disentanglement (DisC-Diff style) | 0.053 [0.04, 0.08] | 0.294 | 0.052 | 0.012 | 60 |
| 5-10 mm | 475 | PLAN stage 1 (seg branch) 300 ep | 0.116 [0.09, 0.15] | 0.348 | 0.109 | 0.045 | 103 |
| 5-10 mm | 475 | PLAN stage 2 (seg + det + cls) 300 ep | 0.101 [0.08, 0.13] | 0.539 | 0.103 | 0.038 | 41 |
| 10-15 mm | 365 | nnU-Net 300 ep | 0.425 [0.37, 0.48] | 0.791 | 0.314 | 0.237 | 41 |
| 10-15 mm | 365 | Plain 3D U-Net 300 ep | 0.441 [0.39, 0.49] | 0.785 | 0.322 | 0.242 | 44 |
| 10-15 mm | 365 | U-Net + synthetic small tumours | 0.616 [0.57, 0.66] | 0.703 | 0.415 | 0.363 | 95 |
| 10-15 mm | 365 | U-Net + CC-DiceCE (w = 0.5) | 0.529 [0.48, 0.58] | 0.739 | 0.404 | 0.292 | 68 |
| 10-15 mm | 365 | U-Net + phase disentanglement (DisC-Diff style) | 0.405 [0.36, 0.46] | 0.771 | 0.308 | 0.231 | 44 |
| 10-15 mm | 365 | PLAN stage 1 (seg branch) 300 ep | 0.512 [0.46, 0.56] | 0.760 | 0.371 | 0.280 | 59 |
| 10-15 mm | 365 | PLAN stage 2 (seg + det + cls) 300 ep | 0.488 [0.44, 0.54] | 0.805 | 0.355 | 0.276 | 43 |
| > 15 mm | 742 | nnU-Net 300 ep | 0.871 [0.84, 0.89] | 0.924 | 0.797 | 0.646 | 53 |
| > 15 mm | 742 | Plain 3D U-Net 300 ep | 0.863 [0.84, 0.89] | 0.932 | 0.830 | 0.640 | 47 |
| > 15 mm | 742 | U-Net + synthetic small tumours | 0.888 [0.86, 0.91] | 0.939 | 0.834 | 0.660 | 43 |
| > 15 mm | 742 | U-Net + CC-DiceCE (w = 0.5) | 0.898 [0.87, 0.92] | 0.926 | 0.836 | 0.660 | 53 |
| > 15 mm | 742 | U-Net + phase disentanglement (DisC-Diff style) | 0.869 [0.84, 0.89] | 0.938 | 0.820 | 0.647 | 43 |
| > 15 mm | 742 | PLAN stage 1 (seg branch) 300 ep | 0.906 [0.88, 0.92] | 0.946 | 0.803 | 0.668 | 38 |
| > 15 mm | 742 | PLAN stage 2 (seg + det + cls) 300 ep | 0.894 [0.87, 0.91] | 0.948 | 0.779 | 0.677 | 36 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 781 | nnU-Net 300 ep | 0.228 [0.20, 0.26] | 0.650 | 0.260 | 0.117 | 96 |
| **5-15 mm** | 781 | Plain 3D U-Net 300 ep | 0.224 [0.20, 0.25] | 0.670 | 0.267 | 0.115 | 86 |
| **5-15 mm** | 781 | U-Net + synthetic small tumours | 0.392 [0.36, 0.43] | 0.503 | 0.361 | 0.217 | 302 |
| **5-15 mm** | 781 | U-Net + CC-DiceCE (w = 0.5) | 0.283 [0.25, 0.32] | 0.641 | 0.337 | 0.146 | 124 |
| **5-15 mm** | 781 | U-Net + phase disentanglement (DisC-Diff style) | 0.201 [0.17, 0.23] | 0.651 | 0.249 | 0.104 | 84 |
| **5-15 mm** | 781 | PLAN stage 1 (seg branch) 300 ep | 0.286 [0.25, 0.32] | 0.625 | 0.317 | 0.145 | 134 |
| **5-15 mm** | 781 | PLAN stage 2 (seg + det + cls) 300 ep | 0.268 [0.24, 0.30] | 0.733 | 0.307 | 0.143 | 76 |
| 0-5 mm (exploratory) | 508 | nnU-Net 300 ep | 0.049 [0.03, 0.07] | 0.309 | 0.048 | 0.000 | 56 |
| 0-5 mm (exploratory) | 508 | Plain 3D U-Net 300 ep | 0.063 [0.04, 0.09] | 0.421 | 0.051 | 0.001 | 44 |
| 0-5 mm (exploratory) | 508 | U-Net + synthetic small tumours | 0.122 [0.10, 0.15] | 0.218 | 0.076 | 0.011 | 223 |
| 0-5 mm (exploratory) | 508 | U-Net + CC-DiceCE (w = 0.5) | 0.081 [0.06, 0.11] | 0.461 | 0.062 | 0.001 | 48 |
| 0-5 mm (exploratory) | 508 | U-Net + phase disentanglement (DisC-Diff style) | 0.059 [0.04, 0.08] | 0.330 | 0.038 | 0.000 | 61 |
| 0-5 mm (exploratory) | 508 | PLAN stage 1 (seg branch) 300 ep | 0.071 [0.05, 0.10] | 0.507 | 0.060 | 0.001 | 35 |
| 0-5 mm (exploratory) | 508 | PLAN stage 2 (seg + det + cls) 300 ep | 0.061 [0.04, 0.09] | 0.564 | 0.054 | 0.001 | 24 |
| 5-10 mm | 458 | nnU-Net 300 ep | 0.079 [0.06, 0.11] | 0.375 | 0.068 | 0.028 | 60 |
| 5-10 mm | 458 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.346 | 0.059 | 0.020 | 53 |
| 5-10 mm | 458 | U-Net + synthetic small tumours | 0.229 [0.19, 0.27] | 0.316 | 0.161 | 0.111 | 227 |
| 5-10 mm | 458 | U-Net + CC-DiceCE (w = 0.5) | 0.100 [0.08, 0.13] | 0.407 | 0.095 | 0.039 | 67 |
| 5-10 mm | 458 | U-Net + phase disentanglement (DisC-Diff style) | 0.052 [0.04, 0.08] | 0.316 | 0.051 | 0.013 | 52 |
| 5-10 mm | 458 | PLAN stage 1 (seg branch) 300 ep | 0.116 [0.09, 0.15] | 0.384 | 0.110 | 0.045 | 85 |
| 5-10 mm | 458 | PLAN stage 2 (seg + det + cls) 300 ep | 0.100 [0.08, 0.13] | 0.535 | 0.101 | 0.039 | 40 |
| 10-15 mm | 323 | nnU-Net 300 ep | 0.440 [0.39, 0.49] | 0.798 | 0.318 | 0.244 | 36 |
| 10-15 mm | 323 | Plain 3D U-Net 300 ep | 0.455 [0.40, 0.51] | 0.817 | 0.330 | 0.249 | 33 |
| 10-15 mm | 323 | U-Net + synthetic small tumours | 0.622 [0.57, 0.67] | 0.728 | 0.421 | 0.368 | 75 |
| 10-15 mm | 323 | U-Net + CC-DiceCE (w = 0.5) | 0.542 [0.49, 0.60] | 0.754 | 0.410 | 0.298 | 57 |
| 10-15 mm | 323 | U-Net + phase disentanglement (DisC-Diff style) | 0.412 [0.36, 0.47] | 0.806 | 0.309 | 0.234 | 32 |
| 10-15 mm | 323 | PLAN stage 1 (seg branch) 300 ep | 0.526 [0.47, 0.58] | 0.776 | 0.379 | 0.288 | 49 |
| 10-15 mm | 323 | PLAN stage 2 (seg + det + cls) 300 ep | 0.505 [0.45, 0.56] | 0.819 | 0.369 | 0.290 | 36 |
| > 15 mm | 645 | nnU-Net 300 ep | 0.878 [0.85, 0.90] | 0.932 | 0.802 | 0.651 | 41 |
| > 15 mm | 645 | Plain 3D U-Net 300 ep | 0.865 [0.84, 0.89] | 0.938 | 0.830 | 0.643 | 37 |
| > 15 mm | 645 | U-Net + synthetic small tumours | 0.893 [0.87, 0.91] | 0.943 | 0.833 | 0.666 | 35 |
| > 15 mm | 645 | U-Net + CC-DiceCE (w = 0.5) | 0.902 [0.88, 0.92] | 0.937 | 0.834 | 0.665 | 39 |
| > 15 mm | 645 | U-Net + phase disentanglement (DisC-Diff style) | 0.873 [0.84, 0.90] | 0.943 | 0.817 | 0.651 | 34 |
| > 15 mm | 645 | PLAN stage 1 (seg branch) 300 ep | 0.909 [0.88, 0.93] | 0.959 | 0.801 | 0.672 | 25 |
| > 15 mm | 645 | PLAN stage 2 (seg + det + cls) 300 ep | 0.898 [0.87, 0.92] | 0.954 | 0.781 | 0.682 | 28 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 32 | nnU-Net 300 ep | 0.312 [0.18, 0.49] | 0.667 | 0.297 | 0.165 | 5 |
| **5-15 mm** | 32 | Plain 3D U-Net 300 ep | 0.344 [0.20, 0.52] | 0.458 | 0.276 | 0.166 | 13 |
| **5-15 mm** | 32 | U-Net + synthetic small tumours | 0.750 [0.58, 0.87] | 0.316 | 0.520 | 0.409 | 52 |
| **5-15 mm** | 32 | U-Net + CC-DiceCE (w = 0.5) | 0.562 [0.39, 0.72] | 0.462 | 0.451 | 0.269 | 21 |
| **5-15 mm** | 32 | U-Net + phase disentanglement (DisC-Diff style) | 0.375 [0.23, 0.55] | 0.522 | 0.342 | 0.197 | 11 |
| **5-15 mm** | 32 | PLAN stage 1 (seg branch) 300 ep | 0.375 [0.23, 0.55] | 0.364 | 0.348 | 0.188 | 21 |
| **5-15 mm** | 32 | PLAN stage 2 (seg + det + cls) 300 ep | 0.312 [0.18, 0.49] | 0.769 | 0.235 | 0.123 | 3 |
| 0-5 mm (exploratory) | 1 | nnU-Net 300 ep | 1.000 [0.21, 1.00] | 0.091 | 1.000 | 0.000 | 10 |
| 0-5 mm (exploratory) | 1 | Plain 3D U-Net 300 ep | 1.000 [0.21, 1.00] | 0.067 | 1.000 | 0.000 | 14 |
| 0-5 mm (exploratory) | 1 | U-Net + synthetic small tumours | 1.000 [0.21, 1.00] | 0.011 | 1.000 | 0.000 | 93 |
| 0-5 mm (exploratory) | 1 | U-Net + CC-DiceCE (w = 0.5) | 1.000 [0.21, 1.00] | 0.077 | 1.000 | 0.000 | 12 |
| 0-5 mm (exploratory) | 1 | U-Net + phase disentanglement (DisC-Diff style) | 1.000 [0.21, 1.00] | 0.056 | 1.000 | 0.000 | 17 |
| 0-5 mm (exploratory) | 1 | PLAN stage 1 (seg branch) 300 ep | 1.000 [0.21, 1.00] | 0.250 | 1.000 | 0.000 | 3 |
| 0-5 mm (exploratory) | 1 | PLAN stage 2 (seg + det + cls) 300 ep | 1.000 [0.21, 1.00] | 0.500 | 1.000 | 0.000 | 1 |
| 5-10 mm | 9 | nnU-Net 300 ep | 0.111 [0.02, 0.44] | 0.333 | 0.061 | 0.000 | 2 |
| 5-10 mm | 9 | Plain 3D U-Net 300 ep | 0.111 [0.02, 0.44] | 0.143 | 0.087 | 0.000 | 6 |
| 5-10 mm | 9 | U-Net + synthetic small tumours | 0.889 [0.56, 0.98] | 0.174 | 0.508 | 0.382 | 38 |
| 5-10 mm | 9 | U-Net + CC-DiceCE (w = 0.5) | 0.444 [0.19, 0.73] | 0.250 | 0.222 | 0.113 | 12 |
| 5-10 mm | 9 | U-Net + phase disentanglement (DisC-Diff style) | 0.111 [0.02, 0.44] | 0.143 | 0.130 | 0.000 | 6 |
| 5-10 mm | 9 | PLAN stage 1 (seg branch) 300 ep | 0.111 [0.02, 0.44] | 0.067 | 0.117 | 0.000 | 14 |
| 5-10 mm | 9 | PLAN stage 2 (seg + det + cls) 300 ep | 0.111 [0.02, 0.44] | 1.000 | 0.131 | 0.000 | 0 |
| 10-15 mm | 23 | nnU-Net 300 ep | 0.391 [0.22, 0.59] | 0.750 | 0.333 | 0.229 | 3 |
| 10-15 mm | 23 | Plain 3D U-Net 300 ep | 0.435 [0.26, 0.63] | 0.588 | 0.306 | 0.232 | 7 |
| 10-15 mm | 23 | U-Net + synthetic small tumours | 0.696 [0.49, 0.84] | 0.533 | 0.522 | 0.420 | 14 |
| 10-15 mm | 23 | U-Net + CC-DiceCE (w = 0.5) | 0.609 [0.41, 0.78] | 0.609 | 0.486 | 0.330 | 9 |
| 10-15 mm | 23 | U-Net + phase disentanglement (DisC-Diff style) | 0.478 [0.29, 0.67] | 0.688 | 0.375 | 0.274 | 5 |
| 10-15 mm | 23 | PLAN stage 1 (seg branch) 300 ep | 0.478 [0.29, 0.67] | 0.611 | 0.384 | 0.261 | 7 |
| 10-15 mm | 23 | PLAN stage 2 (seg + det + cls) 300 ep | 0.391 [0.22, 0.59] | 0.750 | 0.251 | 0.170 | 3 |
| > 15 mm | 49 | nnU-Net 300 ep | 0.837 [0.71, 0.91] | 0.854 | 0.764 | 0.564 | 7 |
| > 15 mm | 49 | Plain 3D U-Net 300 ep | 0.857 [0.73, 0.93] | 0.913 | 0.836 | 0.566 | 4 |
| > 15 mm | 49 | U-Net + synthetic small tumours | 0.837 [0.71, 0.91] | 0.911 | 0.845 | 0.561 | 4 |
| > 15 mm | 49 | U-Net + CC-DiceCE (w = 0.5) | 0.857 [0.73, 0.93] | 0.894 | 0.851 | 0.561 | 5 |
| > 15 mm | 49 | U-Net + phase disentanglement (DisC-Diff style) | 0.857 [0.73, 0.93] | 0.913 | 0.841 | 0.571 | 4 |
| > 15 mm | 49 | PLAN stage 1 (seg branch) 300 ep | 0.878 [0.76, 0.94] | 0.878 | 0.803 | 0.600 | 6 |
| > 15 mm | 49 | PLAN stage 2 (seg + det + cls) 300 ep | 0.796 [0.66, 0.89] | 0.929 | 0.753 | 0.559 | 3 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 27 | nnU-Net 300 ep | 0.148 [0.06, 0.32] | 0.500 | 0.202 | 0.099 | 4 |
| **5-15 mm** | 27 | Plain 3D U-Net 300 ep | 0.148 [0.06, 0.32] | 0.308 | 0.194 | 0.094 | 9 |
| **5-15 mm** | 27 | U-Net + synthetic small tumours | 0.370 [0.22, 0.56] | 0.286 | 0.195 | 0.183 | 25 |
| **5-15 mm** | 27 | U-Net + CC-DiceCE (w = 0.5) | 0.148 [0.06, 0.32] | 0.364 | 0.193 | 0.092 | 7 |
| **5-15 mm** | 27 | U-Net + phase disentanglement (DisC-Diff style) | 0.148 [0.06, 0.32] | 0.308 | 0.186 | 0.093 | 9 |
| **5-15 mm** | 27 | PLAN stage 1 (seg branch) 300 ep | 0.259 [0.13, 0.45] | 0.500 | 0.213 | 0.139 | 7 |
| **5-15 mm** | 27 | PLAN stage 2 (seg + det + cls) 300 ep | 0.259 [0.13, 0.45] | 0.583 | 0.239 | 0.134 | 5 |
| 0-5 mm (exploratory) | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 22 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm (exploratory) | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 8 | nnU-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 8 | Plain 3D U-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 8 | U-Net + synthetic small tumours | 0.250 [0.07, 0.59] | 0.095 | 0.114 | 0.105 | 19 |
| 5-10 mm | 8 | U-Net + CC-DiceCE (w = 0.5) | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 8 | U-Net + phase disentanglement (DisC-Diff style) | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 8 | PLAN stage 1 (seg branch) 300 ep | 0.125 [0.02, 0.47] | 0.200 | 0.082 | 0.073 | 4 |
| 5-10 mm | 8 | PLAN stage 2 (seg + det + cls) 300 ep | 0.125 [0.02, 0.47] | 0.500 | 0.146 | 0.052 | 1 |
| 10-15 mm | 19 | nnU-Net 300 ep | 0.211 [0.09, 0.43] | 0.667 | 0.227 | 0.141 | 2 |
| 10-15 mm | 19 | Plain 3D U-Net 300 ep | 0.211 [0.09, 0.43] | 0.500 | 0.219 | 0.134 | 4 |
| 10-15 mm | 19 | U-Net + synthetic small tumours | 0.421 [0.23, 0.64] | 0.571 | 0.206 | 0.215 | 6 |
| 10-15 mm | 19 | U-Net + CC-DiceCE (w = 0.5) | 0.211 [0.09, 0.43] | 0.667 | 0.218 | 0.131 | 2 |
| 10-15 mm | 19 | U-Net + phase disentanglement (DisC-Diff style) | 0.211 [0.09, 0.43] | 0.364 | 0.209 | 0.133 | 7 |
| 10-15 mm | 19 | PLAN stage 1 (seg branch) 300 ep | 0.316 [0.15, 0.54] | 0.667 | 0.230 | 0.167 | 3 |
| 10-15 mm | 19 | PLAN stage 2 (seg + det + cls) 300 ep | 0.316 [0.15, 0.54] | 0.600 | 0.251 | 0.169 | 4 |
| > 15 mm | 48 | nnU-Net 300 ep | 0.812 [0.68, 0.90] | 0.886 | 0.810 | 0.655 | 5 |
| > 15 mm | 48 | Plain 3D U-Net 300 ep | 0.833 [0.70, 0.91] | 0.870 | 0.813 | 0.675 | 6 |
| > 15 mm | 48 | U-Net + synthetic small tumours | 0.875 [0.75, 0.94] | 0.913 | 0.831 | 0.690 | 4 |
| > 15 mm | 48 | U-Net + CC-DiceCE (w = 0.5) | 0.875 [0.75, 0.94] | 0.824 | 0.829 | 0.688 | 9 |
| > 15 mm | 48 | U-Net + phase disentanglement (DisC-Diff style) | 0.833 [0.70, 0.91] | 0.889 | 0.811 | 0.668 | 5 |
| > 15 mm | 48 | PLAN stage 1 (seg branch) 300 ep | 0.896 [0.78, 0.95] | 0.860 | 0.840 | 0.688 | 7 |
| > 15 mm | 48 | PLAN stage 2 (seg + det + cls) 300 ep | 0.938 [0.83, 0.98] | 0.900 | 0.820 | 0.730 | 5 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + synthetic small tumours: {"epochs_logged": 300, "mean_epoch_seconds": 48.1, "final_val_patch_dice_liver_tumour": [0.9574, 0.8441], "best_val_patch_dice_tumour": 0.8728, "final_train_loss": 0.2705, "synthetic_lesions_per_epoch_mean": 496.0}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net + phase disentanglement (DisC-Diff style): {"epochs_logged": 300, "mean_epoch_seconds": 140.8, "final_val_patch_dice_liver_tumour": [0.9579, 0.8644], "best_val_patch_dice_tumour": 0.8827, "final_train_loss": 0.3061, "synthetic_lesions_per_epoch_mean": 0.0, "final_aux_loss": 0.0}
PLAN stage 1 (seg branch) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.4165, "final_val_loss": -1.2835, "final_global_dice_tumour_liver": [0.8185, 0.9638], "final_online_lesion_sens_prec": [0.871, 0.4426], "training_hours": 21.7}
PLAN stage 2 (seg + det + cls) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 4.3773, "final_val_loss": 5.4606, "final_global_dice_tumour_liver": [0.8544, 0.9649], "final_online_lesion_sens_prec": [0.8256, 0.6174], "training_hours": 21.8}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [183, 218, 246], "current_spacing": [1.0, 1.0, 1.0], "do_dummy_2D_data_aug": false, "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | nnU-Net 300 ep | Plain 3D U-Net 300 ep | U-Net + synthetic small tumours | U-Net + CC-DiceCE (w = 0.5) | U-Net + phase disentanglement (DisC-Diff style) | PLAN stage 1 (seg branch) 300 ep | PLAN stage 2 (seg + det + cls) 300 ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 67 | case tumour Dice (mean) | 0.713 | 0.702 | 0.652 | 0.670 | 0.680 | 0.696 | 0.717 |
|  |  | voxel Dice (pooled) | 0.807 | 0.784 | 0.786 | 0.782 | 0.790 | 0.806 | 0.813 |
|  |  | voxel recall (pooled) | 0.757 | 0.753 | 0.772 | 0.764 | 0.756 | 0.773 | 0.764 |
|  |  | voxel precision (pooled) | 0.865 | 0.818 | 0.801 | 0.802 | 0.829 | 0.842 | 0.869 |
|  |  | lesion recall = sensitivity | 0.973 | 0.959 | 0.959 | 0.932 | 0.932 | 0.959 | 0.946 |
|  |  | lesion precision | 0.626 | 0.617 | 0.314 | 0.473 | 0.683 | 0.607 | 0.778 |
|  |  | lesion F1 | 0.762 | 0.751 | 0.473 | 0.627 | 0.789 | 0.743 | 0.854 |
|  |  | false positives per case | 0.64 | 0.66 | 2.31 | 1.15 | 0.48 | 0.69 | 0.30 |
| MCT-LTDiag | 19 | case tumour Dice (mean) | 0.788 | 0.788 | 0.768 | 0.787 | 0.807 | 0.798 | 0.795 |
|  |  | voxel Dice (pooled) | 0.863 | 0.859 | 0.835 | 0.866 | 0.869 | 0.871 | 0.877 |
|  |  | voxel recall (pooled) | 0.854 | 0.851 | 0.818 | 0.883 | 0.874 | 0.871 | 0.867 |
|  |  | voxel precision (pooled) | 0.872 | 0.868 | 0.854 | 0.850 | 0.864 | 0.871 | 0.888 |
|  |  | lesion recall = sensitivity | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 | 0.960 |
|  |  | lesion precision | 0.750 | 0.774 | 0.393 | 0.706 | 0.774 | 0.706 | 0.800 |
|  |  | lesion F1 | 0.842 | 0.857 | 0.558 | 0.814 | 0.857 | 0.814 | 0.873 |
|  |  | false positives per case | 0.42 | 0.37 | 1.95 | 0.53 | 0.37 | 0.53 | 0.32 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.642 | 0.614 | 0.534 | 0.571 | 0.574 | 0.601 | 0.643 |
|  |  | voxel Dice (pooled) | 0.797 | 0.751 | 0.760 | 0.744 | 0.754 | 0.780 | 0.789 |
|  |  | voxel recall (pooled) | 0.742 | 0.734 | 0.769 | 0.738 | 0.724 | 0.752 | 0.739 |
|  |  | voxel precision (pooled) | 0.861 | 0.770 | 0.751 | 0.751 | 0.787 | 0.811 | 0.846 |
|  |  | lesion recall = sensitivity | 0.963 | 0.926 | 0.926 | 0.889 | 0.889 | 0.926 | 0.889 |
|  |  | lesion precision | 0.553 | 0.446 | 0.202 | 0.282 | 0.533 | 0.532 | 0.727 |
|  |  | lesion F1 | 0.703 | 0.602 | 0.331 | 0.429 | 0.667 | 0.676 | 0.800 |
|  |  | false positives per case | 0.62 | 0.91 | 2.91 | 1.79 | 0.62 | 0.65 | 0.26 |
| WAW-TACE | 14 | case tumour Dice (mean) | 0.747 | 0.757 | 0.749 | 0.707 | 0.719 | 0.748 | 0.745 |
|  |  | voxel Dice (pooled) | 0.788 | 0.817 | 0.824 | 0.822 | 0.826 | 0.825 | 0.827 |
|  |  | voxel recall (pooled) | 0.721 | 0.729 | 0.745 | 0.742 | 0.749 | 0.755 | 0.753 |
|  |  | voxel precision (pooled) | 0.868 | 0.929 | 0.922 | 0.922 | 0.920 | 0.910 | 0.918 |
|  |  | lesion recall = sensitivity | 1.000 | 1.000 | 1.000 | 0.955 | 0.955 | 1.000 | 1.000 |
|  |  | lesion precision | 0.611 | 0.786 | 0.537 | 0.778 | 0.840 | 0.611 | 0.815 |
|  |  | lesion F1 | 0.759 | 0.880 | 0.698 | 0.857 | 0.894 | 0.759 | 0.898 |
|  |  | false positives per case | 1.00 | 0.43 | 1.36 | 0.43 | 0.29 | 1.00 | 0.36 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 18 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 71 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 26 |
| 0-5 mm (exploratory) | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 9 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 10 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 54 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 18 |
| 5-10 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 11 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 22 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 21 |
| 10-15 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 11 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| > 15 mm | 74 | nnU-Net 300 ep | 0.973 [0.91, 0.99] | 0.889 | 0.757 | 0.727 | 9 |
| > 15 mm | 74 | Plain 3D U-Net 300 ep | 0.959 [0.89, 0.99] | 0.910 | 0.753 | 0.726 | 7 |
| > 15 mm | 74 | U-Net + synthetic small tumours | 0.959 [0.89, 0.99] | 0.899 | 0.772 | 0.705 | 8 |
| > 15 mm | 74 | U-Net + CC-DiceCE (w = 0.5) | 0.932 [0.85, 0.97] | 0.852 | 0.764 | 0.708 | 12 |
| > 15 mm | 74 | U-Net + phase disentanglement (DisC-Diff style) | 0.932 [0.85, 0.97] | 0.896 | 0.756 | 0.697 | 8 |
| > 15 mm | 74 | PLAN stage 1 (seg branch) 300 ep | 0.959 [0.89, 0.99] | 0.835 | 0.773 | 0.727 | 14 |
| > 15 mm | 74 | PLAN stage 2 (seg + det + cls) 300 ep | 0.946 [0.87, 0.98] | 0.897 | 0.764 | 0.718 | 8 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 0-5 mm (exploratory) | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | n/a | n/a | n/a | 0 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | n/a | n/a | n/a | 0 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 14 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | n/a | n/a | n/a | 0 |
| 5-10 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 6 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | n/a | n/a | n/a | 0 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 8 |
| 10-15 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 2 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| > 15 mm | 25 | nnU-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.854 | 0.772 | 0 |
| > 15 mm | 25 | Plain 3D U-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.851 | 0.776 | 0 |
| > 15 mm | 25 | U-Net + synthetic small tumours | 0.960 [0.80, 0.99] | 0.960 | 0.818 | 0.768 | 1 |
| > 15 mm | 25 | U-Net + CC-DiceCE (w = 0.5) | 0.960 [0.80, 0.99] | 0.923 | 0.883 | 0.783 | 2 |
| > 15 mm | 25 | U-Net + phase disentanglement (DisC-Diff style) | 0.960 [0.80, 0.99] | 0.960 | 0.874 | 0.792 | 1 |
| > 15 mm | 25 | PLAN stage 1 (seg branch) 300 ep | 0.960 [0.80, 0.99] | 0.960 | 0.871 | 0.784 | 1 |
| > 15 mm | 25 | PLAN stage 2 (seg + det + cls) 300 ep | 0.960 [0.80, 0.99] | 0.960 | 0.867 | 0.784 | 1 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 48 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 24 |
| 0-5 mm (exploratory) | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 9 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 33 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 17 |
| 5-10 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 3 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 12 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 11 |
| 10-15 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 2 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | n/a | n/a | n/a | 0 |
| > 15 mm | 27 | nnU-Net 300 ep | 0.963 [0.82, 0.99] | 0.788 | 0.742 | 0.694 | 7 |
| > 15 mm | 27 | Plain 3D U-Net 300 ep | 0.926 [0.77, 0.98] | 0.781 | 0.734 | 0.681 | 7 |
| > 15 mm | 27 | U-Net + synthetic small tumours | 0.926 [0.77, 0.98] | 0.806 | 0.769 | 0.661 | 6 |
| > 15 mm | 27 | U-Net + CC-DiceCE (w = 0.5) | 0.889 [0.72, 0.96] | 0.727 | 0.738 | 0.648 | 9 |
| > 15 mm | 27 | U-Net + phase disentanglement (DisC-Diff style) | 0.889 [0.72, 0.96] | 0.774 | 0.724 | 0.649 | 7 |
| > 15 mm | 27 | PLAN stage 1 (seg branch) 300 ep | 0.926 [0.77, 0.98] | 0.714 | 0.752 | 0.677 | 10 |
| > 15 mm | 27 | PLAN stage 2 (seg + det + cls) 300 ep | 0.889 [0.72, 0.96] | 0.800 | 0.739 | 0.658 | 6 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| 0-5 mm (exploratory) | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm (exploratory) | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 10 |
| 0-5 mm (exploratory) | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm (exploratory) | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | n/a | n/a | n/a | 0 |
| 0-5 mm (exploratory) | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm (exploratory) | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | n/a | n/a | n/a | 0 |
| 5-10 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 7 |
| 5-10 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 10-15 mm | 0 | nnU-Net 300 ep | n/a | 0.000 | n/a | n/a | 3 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | U-Net + synthetic small tumours | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 0 | U-Net + CC-DiceCE (w = 0.5) | n/a | 0.000 | n/a | n/a | 2 |
| 10-15 mm | 0 | U-Net + phase disentanglement (DisC-Diff style) | n/a | n/a | n/a | n/a | 0 |
| 10-15 mm | 0 | PLAN stage 1 (seg branch) 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | PLAN stage 2 (seg + det + cls) 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| > 15 mm | 22 | nnU-Net 300 ep | 1.000 [0.85, 1.00] | 0.917 | 0.721 | 0.716 | 2 |
| > 15 mm | 22 | Plain 3D U-Net 300 ep | 1.000 [0.85, 1.00] | 1.000 | 0.729 | 0.724 | 0 |
| > 15 mm | 22 | U-Net + synthetic small tumours | 1.000 [0.85, 1.00] | 0.957 | 0.745 | 0.686 | 1 |
| > 15 mm | 22 | U-Net + CC-DiceCE (w = 0.5) | 0.955 [0.78, 0.99] | 0.955 | 0.742 | 0.697 | 1 |
| > 15 mm | 22 | U-Net + phase disentanglement (DisC-Diff style) | 0.955 [0.78, 0.99] | 1.000 | 0.749 | 0.648 | 0 |
| > 15 mm | 22 | PLAN stage 1 (seg branch) 300 ep | 1.000 [0.85, 1.00] | 0.880 | 0.755 | 0.722 | 3 |
| > 15 mm | 22 | PLAN stage 2 (seg + det + cls) 300 ep | 1.000 [0.85, 1.00] | 0.957 | 0.753 | 0.716 | 1 |

### Training

```
nnU-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 56.1, "final_pseudo_dice_liver_tumour": [0.9565, 0.7999], "best_pseudo_dice_tumour": 0.8543}
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
U-Net + synthetic small tumours: {"epochs_logged": 300, "mean_epoch_seconds": 48.1, "final_val_patch_dice_liver_tumour": [0.9574, 0.8441], "best_val_patch_dice_tumour": 0.8728, "final_train_loss": 0.2705, "synthetic_lesions_per_epoch_mean": 496.0}
U-Net + CC-DiceCE (w = 0.5): {"epochs_logged": 300, "mean_epoch_seconds": 60.6, "final_val_patch_dice_liver_tumour": [0.9563, 0.8452], "best_val_patch_dice_tumour": 0.8837, "final_train_loss": 0.3527, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net + phase disentanglement (DisC-Diff style): {"epochs_logged": 300, "mean_epoch_seconds": 140.8, "final_val_patch_dice_liver_tumour": [0.9579, 0.8644], "best_val_patch_dice_tumour": 0.8827, "final_train_loss": 0.3061, "synthetic_lesions_per_epoch_mean": 0.0, "final_aux_loss": 0.0}
PLAN stage 1 (seg branch) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": -1.4165, "final_val_loss": -1.2835, "final_global_dice_tumour_liver": [0.8185, 0.9638], "final_online_lesion_sens_prec": [0.871, 0.4426], "training_hours": 21.7}
PLAN stage 2 (seg + det + cls) 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": null, "final_train_loss": 4.3773, "final_val_loss": 5.4606, "final_global_dice_tumour_liver": [0.8544, 0.9649], "final_online_lesion_sens_prec": [0.8256, 0.6174], "training_hours": 21.8}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
PLAN nnU-Net v1 plans: {"0": {"batch_size": 2, "num_pool_per_axis": [4, 5, 5], "patch_size": [112, 128, 160], "median_patient_size_in_voxels": [183, 218, 246], "current_spacing": [1.0, 1.0, 1.0], "do_dummy_2D_data_aug": false, "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}, "base_num_features": 32, "num_modalities": 4, "num_classes": 2, "plan_num_patches": 60}
```

## Artefacts

`results/2026-09-30_evaluation_update_5_15mm_primary/`: ccdicece_config.json, ccdicece_history.json, ccdicece_jobs.json, ccdicece_test_per_lesion.csv, ccdicece_test_size_binned_metrics.csv, ccdicece_val_per_lesion.csv, ccdicece_val_size_binned_metrics.csv, disentangled_config.json, disentangled_history.json, disentangled_jobs.json, disentangled_test_per_lesion.csv, disentangled_test_size_binned_metrics.csv, disentangled_val_per_lesion.csv, disentangled_val_size_binned_metrics.csv, nnunet_cohorts.json, nnunet_nnUNetPlans.json, nnunet_test_per_lesion.csv, nnunet_test_size_binned_metrics.csv, nnunet_training_log_2026_9_27_02_50_22.txt, nnunet_val_per_lesion.csv, nnunet_val_size_binned_metrics.csv, plan1_jobs.json, plan1_plans_summary.json, plan1_test_per_lesion.csv, plan1_test_size_binned_metrics.csv, plan1_training_log_2026_9_27_22_13_27.txt, plan1_training_log_2026_9_28_19_52_43.txt, plan1_training_log_2026_9_28_20_00_48.txt, plan1_val_per_lesion.csv, plan1_val_size_binned_metrics.csv, plan2_jobs.json, plan2_plans_summary.json, plan2_test_per_lesion.csv, plan2_test_size_binned_metrics.csv, plan2_training_log_2026_9_28_21_11_28.txt, plan2_training_log_2026_9_29_18_59_52.txt, plan2_training_log_2026_9_29_19_10_58.txt, plan2_val_per_lesion.csv, plan2_val_size_binned_metrics.csv, synth_config.json, synth_history.json, synth_jobs.json, synth_test_per_lesion.csv, synth_test_size_binned_metrics.csv, synth_val_per_lesion.csv, synth_val_size_binned_metrics.csv, unet3d_config.json, unet3d_history.json, unet3d_test_per_lesion.csv, unet3d_test_size_binned_metrics.csv, unet3d_val_per_lesion.csv, unet3d_val_size_binned_metrics.csv

