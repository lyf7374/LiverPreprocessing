# 2026-10-03: postprocessing diagnostic stratified

**Purpose.** Diagnostic follow-up to the lesion-sampling record: did synthetic small tumours and CC-DiceCE really fail on the stratified split, or only their argmax operating points? Checks (1) the loss implementations, (2) what the extra false positives are, (3) whether a minimum predicted-component size restores usable operating points and changes the ranking at matched false-positive rates. No new training.

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: stratified seed 0 (train 727 / val 105 / test 210; lesions of every size in all subsets); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

(1) Regression check: with no ignored voxels the ignore-aware DiceCE and CC-DiceCE equal the previous implementations to 1e-6 (same loss as in experiment 2). (2) False-positive components per size bin from the FROC tables at threshold 0.5 (test subset, 210 cases). (3) evaluate_by_lesion_size.py --min-pred-voxels 20 / 50 / 100 on the argmax label maps (components below the size are dropped before matching), and froc_by_lesion_size.py --min-pred-voxels 33 (4 mm equivalent diameter, i.e. below the 5 mm lower edge of the target bin) on the probability maps of the three lesion-sampling runs; CPU jobs 29842798 and 29842807-29842809.

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `stratified` = Plain 3D U-Net, stratified split (small lesions in training); `strat_lesion` = U-Net, lesion-balanced sampling (stratified split); `strat_lesion_synth` = U-Net, lesion sampling + synthetic tumours (stratified); `strat_lesion_synth_cc` = U-Net, lesion sampling + synthetic + CC-DiceCE (stratified)

**Findings.**

What the extra false positives are: at threshold 0.5 the false-positive components in the 0-5 mm bin are 51 (lesion sampling), 239 (+ synthetic) and 1178 (+ synthetic + CC-DiceCE) versus 60 / 134 / 256 in the 5-10 mm bin and 34 / 43 / 54 in the 10-15 mm bin; they concentrate on PLC-CECT and WAW-TACE (CC run 10.2 and 10.5 per case versus 4.3 on MCT-LTDiag), the datasets with almost no real small lesions in training and heterogeneous cirrhotic livers. So the collapse is a flood of confident tiny components below the target size, not a loss of sensitivity. Minimum component size on the argmax maps (FP per case / lesion recall / 5-10 mm / 10-15 mm sensitivity): + synthetic at 100 voxels 0.83 / 0.679 / 0.279 / 0.662 (unfiltered 2.25 / 0.696 / 0.361 / 0.662); + CC-DiceCE at 100 voxels 1.28 / 0.716 / 0.459 / 0.750 (unfiltered 7.43 / 0.742 / 0.525 / 0.765); lesion sampling at 20 voxels 0.76 / 0.659 / 0.230 / 0.618. FROC with components below 33 voxels removed, 5-15 mm sensitivity at 0.5 / 1 / 2 / 4 FP per case: lesion sampling 0.349 / 0.465 / 0.554 / n.a. (operating range 0.27-3.9 FP per case, up from 0.447 / 0.533 unfiltered); + synthetic n.a. / 0.457 / 0.554 / 0.642 (range 0.63-4.8; the 1 FP point becomes reachable); + CC-DiceCE n.a. / n.a. / 0.631 / 0.667 (range 1.1-7.4). In the 5-10 mm bin at 2 FP per case: 0.396 / 0.422 / 0.492. Conclusions: (a) no implementation error; (b) a size filter matched to the target (components < 4 mm removed) is a necessary standard post-processing step for every model and improves even the lesion-sampling run (+2 points at 1-2 FP per case); (c) with the filter, synthesis is equal to lesion sampling at 1-2 FP per case and better only above 2 FP per case; (d) CC-DiceCE on top of synthesis gives the most sensitive curve above 2 FP per case (0.631 vs 0.554 at 2 FP, 0.492 vs 0.396 in 5-10 mm) but cannot operate below about 1 FP per case even with the filter, so it is the natural candidate-generation stage for a two-stage detector while lesion sampling alone is the better single-stage model at <= 1 FP per case. Revised recipe: lesion-balanced sampling + fragment masking + minimum component size 33 voxels as the default; synthesis and CC-DiceCE kept for the high-recall first stage of the planned two-stage model.

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | Plain 3D U-Net, stratified split (small lesions in training) | U-Net, lesion-balanced sampling (stratified split) | U-Net, lesion sampling + synthetic tumours (stratified) | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) |
|---|---:|---|---:|---:|---:|---:|
| overall | 210 | case tumour Dice (mean) | 0.703 | 0.696 | 0.704 | 0.693 |
|  |  | voxel Dice (pooled) | 0.856 | 0.853 | 0.859 | 0.856 |
|  |  | voxel recall (pooled) | 0.854 | 0.842 | 0.854 | 0.851 |
|  |  | voxel precision (pooled) | 0.858 | 0.864 | 0.864 | 0.860 |
|  |  | lesion recall = sensitivity | 0.647 | 0.663 | 0.696 | 0.742 |
|  |  | lesion precision | 0.639 | 0.649 | 0.428 | 0.194 |
|  |  | lesion F1 | 0.643 | 0.656 | 0.530 | 0.308 |
|  |  | false positives per case | 0.88 | 0.87 | 2.25 | 7.43 |
| MCT-LTDiag | 106 | case tumour Dice (mean) | 0.756 | 0.755 | 0.755 | 0.759 |
|  |  | voxel Dice (pooled) | 0.895 | 0.894 | 0.894 | 0.893 |
|  |  | voxel recall (pooled) | 0.899 | 0.893 | 0.897 | 0.897 |
|  |  | voxel precision (pooled) | 0.890 | 0.895 | 0.891 | 0.888 |
|  |  | lesion recall = sensitivity | 0.573 | 0.593 | 0.624 | 0.678 |
|  |  | lesion precision | 0.749 | 0.763 | 0.550 | 0.349 |
|  |  | lesion F1 | 0.649 | 0.668 | 0.584 | 0.461 |
|  |  | false positives per case | 0.71 | 0.68 | 1.89 | 4.66 |
| PLC-CECT | 69 | case tumour Dice (mean) | 0.602 | 0.589 | 0.609 | 0.567 |
|  |  | voxel Dice (pooled) | 0.816 | 0.808 | 0.822 | 0.814 |
|  |  | voxel recall (pooled) | 0.779 | 0.758 | 0.783 | 0.774 |
|  |  | voxel precision (pooled) | 0.856 | 0.866 | 0.866 | 0.859 |
|  |  | lesion recall = sensitivity | 0.937 | 0.921 | 0.968 | 0.984 |
|  |  | lesion precision | 0.472 | 0.457 | 0.253 | 0.080 |
|  |  | lesion F1 | 0.628 | 0.611 | 0.401 | 0.148 |
|  |  | false positives per case | 0.96 | 1.00 | 2.61 | 10.35 |
| WAW-TACE | 35 | case tumour Dice (mean) | 0.722 | 0.706 | 0.719 | 0.730 |
|  |  | voxel Dice (pooled) | 0.765 | 0.761 | 0.773 | 0.775 |
|  |  | voxel recall (pooled) | 0.850 | 0.843 | 0.849 | 0.849 |
|  |  | voxel precision (pooled) | 0.695 | 0.694 | 0.710 | 0.712 |
|  |  | lesion recall = sensitivity | 0.849 | 0.868 | 0.906 | 0.925 |
|  |  | lesion precision | 0.506 | 0.529 | 0.343 | 0.122 |
|  |  | lesion F1 | 0.634 | 0.657 | 0.497 | 0.215 |
|  |  | false positives per case | 1.26 | 1.17 | 2.63 | 10.09 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 129 | Plain 3D U-Net, stratified split (small lesions in training) | 0.403 [0.32, 0.49] | 0.354 | 0.346 | 0.205 | 95 |
| **5-15 mm** | 129 | U-Net, lesion-balanced sampling (stratified split) | 0.442 [0.36, 0.53] | 0.373 | 0.358 | 0.212 | 96 |
| **5-15 mm** | 129 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.519 [0.43, 0.60] | 0.271 | 0.392 | 0.269 | 180 |
| **5-15 mm** | 129 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.651 [0.57, 0.73] | 0.207 | 0.469 | 0.351 | 321 |
| 0-5 mm (exploratory) | 91 | Plain 3D U-Net, stratified split (small lesions in training) | 0.077 [0.04, 0.15] | 0.115 | 0.051 | 0.002 | 54 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion-balanced sampling (stratified split) | 0.121 [0.07, 0.20] | 0.175 | 0.067 | 0.005 | 52 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.154 [0.09, 0.24] | 0.051 | 0.137 | 0.031 | 259 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.176 [0.11, 0.27] | 0.013 | 0.170 | 0.045 | 1205 |
| 5-10 mm | 61 | Plain 3D U-Net, stratified split (small lesions in training) | 0.213 [0.13, 0.33] | 0.167 | 0.137 | 0.060 | 65 |
| 5-10 mm | 61 | U-Net, lesion-balanced sampling (stratified split) | 0.246 [0.16, 0.37] | 0.192 | 0.152 | 0.056 | 63 |
| 5-10 mm | 61 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.361 [0.25, 0.49] | 0.138 | 0.239 | 0.144 | 138 |
| 5-10 mm | 61 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.525 [0.40, 0.64] | 0.108 | 0.399 | 0.249 | 265 |
| 10-15 mm | 68 | Plain 3D U-Net, stratified split (small lesions in training) | 0.574 [0.46, 0.68] | 0.565 | 0.390 | 0.335 | 30 |
| 10-15 mm | 68 | U-Net, lesion-balanced sampling (stratified split) | 0.618 [0.50, 0.72] | 0.560 | 0.402 | 0.351 | 33 |
| 10-15 mm | 68 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.662 [0.54, 0.76] | 0.517 | 0.425 | 0.380 | 42 |
| 10-15 mm | 68 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.765 [0.65, 0.85] | 0.481 | 0.483 | 0.443 | 56 |
| > 15 mm | 287 | Plain 3D U-Net, stratified split (small lesions in training) | 0.937 [0.90, 0.96] | 0.882 | 0.855 | 0.709 | 36 |
| > 15 mm | 287 | U-Net, lesion-balanced sampling (stratified split) | 0.934 [0.90, 0.96] | 0.887 | 0.844 | 0.708 | 34 |
| > 15 mm | 287 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.948 [0.92, 0.97] | 0.892 | 0.855 | 0.720 | 33 |
| > 15 mm | 287 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.962 [0.93, 0.98] | 0.887 | 0.852 | 0.730 | 35 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 118 | Plain 3D U-Net, stratified split (small lesions in training) | 0.407 [0.32, 0.50] | 0.516 | 0.341 | 0.204 | 45 |
| **5-15 mm** | 118 | U-Net, lesion-balanced sampling (stratified split) | 0.441 [0.35, 0.53] | 0.571 | 0.351 | 0.207 | 39 |
| **5-15 mm** | 118 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.500 [0.41, 0.59] | 0.410 | 0.370 | 0.251 | 85 |
| **5-15 mm** | 118 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.636 [0.55, 0.72] | 0.322 | 0.451 | 0.338 | 158 |
| 0-5 mm (exploratory) | 91 | Plain 3D U-Net, stratified split (small lesions in training) | 0.077 [0.04, 0.15] | 0.280 | 0.051 | 0.002 | 18 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion-balanced sampling (stratified split) | 0.121 [0.07, 0.20] | 0.367 | 0.067 | 0.005 | 19 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.154 [0.09, 0.24] | 0.120 | 0.137 | 0.031 | 103 |
| 0-5 mm (exploratory) | 91 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.176 [0.11, 0.27] | 0.047 | 0.170 | 0.045 | 325 |
| 5-10 mm | 57 | Plain 3D U-Net, stratified split (small lesions in training) | 0.228 [0.14, 0.35] | 0.283 | 0.148 | 0.064 | 33 |
| 5-10 mm | 57 | U-Net, lesion-balanced sampling (stratified split) | 0.263 [0.17, 0.39] | 0.326 | 0.164 | 0.060 | 31 |
| 5-10 mm | 57 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.351 [0.24, 0.48] | 0.235 | 0.232 | 0.131 | 65 |
| 5-10 mm | 57 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.509 [0.38, 0.63] | 0.181 | 0.390 | 0.237 | 131 |
| 10-15 mm | 61 | Plain 3D U-Net, stratified split (small lesions in training) | 0.574 [0.45, 0.69] | 0.745 | 0.384 | 0.334 | 12 |
| 10-15 mm | 61 | U-Net, lesion-balanced sampling (stratified split) | 0.607 [0.48, 0.72] | 0.822 | 0.392 | 0.344 | 8 |
| 10-15 mm | 61 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.639 [0.51, 0.75] | 0.661 | 0.400 | 0.363 | 20 |
| 10-15 mm | 61 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.754 [0.63, 0.84] | 0.630 | 0.465 | 0.433 | 27 |
| > 15 mm | 182 | Plain 3D U-Net, stratified split (small lesions in training) | 0.929 [0.88, 0.96] | 0.934 | 0.901 | 0.700 | 12 |
| > 15 mm | 182 | U-Net, lesion-balanced sampling (stratified split) | 0.929 [0.88, 0.96] | 0.923 | 0.895 | 0.706 | 14 |
| > 15 mm | 182 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.940 [0.90, 0.97] | 0.934 | 0.899 | 0.712 | 12 |
| > 15 mm | 182 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.956 [0.92, 0.98] | 0.941 | 0.899 | 0.731 | 11 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 5 | Plain 3D U-Net, stratified split (small lesions in training) | 0.600 [0.23, 0.88] | 0.088 | 0.450 | 0.330 | 31 |
| **5-15 mm** | 5 | U-Net, lesion-balanced sampling (stratified split) | 0.600 [0.23, 0.88] | 0.071 | 0.487 | 0.364 | 39 |
| **5-15 mm** | 5 | U-Net, lesion sampling + synthetic tumours (stratified) | 1.000 [0.57, 1.00] | 0.067 | 0.823 | 0.679 | 70 |
| **5-15 mm** | 5 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 1.000 [0.57, 1.00] | 0.041 | 0.860 | 0.692 | 118 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 23 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 19 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | 0.000 | n/a | n/a | 99 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | 0.000 | n/a | n/a | 584 |
| 5-10 mm | 2 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 21 |
| 5-10 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 24 |
| 5-10 mm | 2 | U-Net, lesion sampling + synthetic tumours (stratified) | 1.000 [0.34, 1.00] | 0.034 | 0.577 | 0.660 | 56 |
| 5-10 mm | 2 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 1.000 [0.34, 1.00] | 0.020 | 0.709 | 0.711 | 100 |
| 10-15 mm | 3 | Plain 3D U-Net, stratified split (small lesions in training) | 1.000 [0.44, 1.00] | 0.231 | 0.539 | 0.550 | 10 |
| 10-15 mm | 3 | U-Net, lesion-balanced sampling (stratified split) | 1.000 [0.44, 1.00] | 0.167 | 0.584 | 0.607 | 15 |
| 10-15 mm | 3 | U-Net, lesion sampling + synthetic tumours (stratified) | 1.000 [0.44, 1.00] | 0.176 | 0.871 | 0.691 | 14 |
| 10-15 mm | 3 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 1.000 [0.44, 1.00] | 0.143 | 0.890 | 0.680 | 18 |
| > 15 mm | 58 | Plain 3D U-Net, stratified split (small lesions in training) | 0.966 [0.88, 0.99] | 0.824 | 0.779 | 0.708 | 12 |
| > 15 mm | 58 | U-Net, lesion-balanced sampling (stratified split) | 0.948 [0.86, 0.98] | 0.833 | 0.758 | 0.691 | 11 |
| > 15 mm | 58 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.966 [0.88, 0.99] | 0.836 | 0.783 | 0.714 | 11 |
| > 15 mm | 58 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.983 [0.91, 1.00] | 0.826 | 0.774 | 0.712 | 12 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 6 | Plain 3D U-Net, stratified split (small lesions in training) | 0.167 [0.03, 0.56] | 0.050 | 0.333 | 0.130 | 19 |
| **5-15 mm** | 6 | U-Net, lesion-balanced sampling (stratified split) | 0.333 [0.10, 0.70] | 0.100 | 0.384 | 0.175 | 18 |
| **5-15 mm** | 6 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.500 [0.19, 0.81] | 0.107 | 0.422 | 0.274 | 25 |
| **5-15 mm** | 6 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.667 [0.30, 0.90] | 0.082 | 0.451 | 0.322 | 45 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 13 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 14 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | 0.000 | n/a | n/a | 57 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | 0.000 | n/a | n/a | 296 |
| 5-10 mm | 2 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 11 |
| 5-10 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 8 |
| 5-10 mm | 2 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 17 |
| 5-10 mm | 2 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.500 [0.09, 0.91] | 0.029 | 0.263 | 0.133 | 34 |
| 10-15 mm | 4 | Plain 3D U-Net, stratified split (small lesions in training) | 0.250 [0.05, 0.70] | 0.111 | 0.369 | 0.195 | 8 |
| 10-15 mm | 4 | U-Net, lesion-balanced sampling (stratified split) | 0.500 [0.15, 0.85] | 0.167 | 0.425 | 0.262 | 10 |
| 10-15 mm | 4 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.750 [0.30, 0.95] | 0.273 | 0.467 | 0.411 | 8 |
| 10-15 mm | 4 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.750 [0.30, 0.95] | 0.214 | 0.471 | 0.417 | 11 |
| > 15 mm | 47 | Plain 3D U-Net, stratified split (small lesions in training) | 0.936 [0.83, 0.98] | 0.786 | 0.851 | 0.742 | 12 |
| > 15 mm | 47 | U-Net, lesion-balanced sampling (stratified split) | 0.936 [0.83, 0.98] | 0.830 | 0.843 | 0.737 | 9 |
| > 15 mm | 47 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.957 [0.86, 0.99] | 0.818 | 0.849 | 0.756 | 10 |
| > 15 mm | 47 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.957 [0.86, 0.99] | 0.789 | 0.849 | 0.750 | 12 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.335 | 0.447 | 0.533 | 0.475 |
| 5-15 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | n/a | 0.512 | 0.573 |
| 5-15 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.626 |
| 5-10 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.100 | 0.257 | 0.369 | 0.290 |
| 5-10 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | n/a | 0.346 | 0.448 |
| 5-10 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.494 |
| 10-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.546 | 0.618 | 0.680 | 0.642 |
| 10-15 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | n/a | 0.662 | 0.686 |
| 10-15 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.745 |
| > 15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.890 | 0.938 | 0.965 | 0.941 |
| > 15 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | n/a | 0.931 | 0.957 |
| > 15 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.948 |
| all sizes | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.601 | 0.667 | 0.711 | 0.679 |
| all sizes | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | n/a | 0.683 | 0.725 |
| all sizes | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.725 |
| 0-5 mm (exploratory) | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.066 | 0.121 | 0.159 | 0.142 |
| 0-5 mm (exploratory) | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | n/a | 0.143 | 0.205 |
| 0-5 mm (exploratory) | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.159 |

### Training

```
Plain 3D U-Net, stratified split (small lesions in training): {"epochs_logged": 300, "mean_epoch_seconds": 49.0, "final_val_patch_dice_liver_tumour": [0.9447, 0.8548], "best_val_patch_dice_tumour": 0.8928, "final_train_loss": 0.3325, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net, lesion-balanced sampling (stratified split): {"epochs_logged": 300, "mean_epoch_seconds": 49.1, "final_val_patch_dice_liver_tumour": [0.9582, 0.8351], "best_val_patch_dice_tumour": 0.8921, "final_train_loss": 0.3521, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net, lesion sampling + synthetic tumours (stratified): {"epochs_logged": 300, "mean_epoch_seconds": 50.1, "final_val_patch_dice_liver_tumour": [0.958, 0.8406], "best_val_patch_dice_tumour": 0.891, "final_train_loss": 0.3047, "synthetic_lesions_per_epoch_mean": 492.5}
U-Net, lesion sampling + synthetic + CC-DiceCE (stratified): {"epochs_logged": 300, "mean_epoch_seconds": 67.2, "final_val_patch_dice_liver_tumour": [0.9566, 0.8426], "best_val_patch_dice_tumour": 0.8866, "final_train_loss": 0.3984, "synthetic_lesions_per_epoch_mean": 492.5}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | Plain 3D U-Net, stratified split (small lesions in training) | U-Net, lesion-balanced sampling (stratified split) | U-Net, lesion sampling + synthetic tumours (stratified) | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) |
|---|---:|---|---:|---:|---:|---:|
| overall | 105 | case tumour Dice (mean) | 0.693 | 0.709 | 0.713 | 0.705 |
|  |  | voxel Dice (pooled) | 0.845 | 0.850 | 0.847 | 0.841 |
|  |  | voxel recall (pooled) | 0.864 | 0.866 | 0.866 | 0.862 |
|  |  | voxel precision (pooled) | 0.826 | 0.835 | 0.828 | 0.822 |
|  |  | lesion recall = sensitivity | 0.551 | 0.577 | 0.626 | 0.682 |
|  |  | lesion precision | 0.709 | 0.707 | 0.393 | 0.191 |
|  |  | lesion F1 | 0.620 | 0.635 | 0.483 | 0.298 |
|  |  | false positives per case | 0.66 | 0.70 | 2.81 | 8.39 |
| MCT-LTDiag | 54 | case tumour Dice (mean) | 0.763 | 0.763 | 0.762 | 0.759 |
|  |  | voxel Dice (pooled) | 0.869 | 0.872 | 0.865 | 0.864 |
|  |  | voxel recall (pooled) | 0.867 | 0.870 | 0.865 | 0.870 |
|  |  | voxel precision (pooled) | 0.871 | 0.874 | 0.866 | 0.859 |
|  |  | lesion recall = sensitivity | 0.486 | 0.510 | 0.569 | 0.644 |
|  |  | lesion precision | 0.783 | 0.777 | 0.558 | 0.370 |
|  |  | lesion F1 | 0.600 | 0.616 | 0.564 | 0.470 |
|  |  | false positives per case | 0.63 | 0.69 | 2.11 | 5.13 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.589 | 0.642 | 0.663 | 0.642 |
|  |  | voxel Dice (pooled) | 0.821 | 0.829 | 0.828 | 0.821 |
|  |  | voxel recall (pooled) | 0.865 | 0.869 | 0.871 | 0.862 |
|  |  | voxel precision (pooled) | 0.782 | 0.792 | 0.788 | 0.784 |
|  |  | lesion recall = sensitivity | 0.893 | 0.964 | 0.964 | 0.893 |
|  |  | lesion precision | 0.581 | 0.562 | 0.159 | 0.053 |
|  |  | lesion F1 | 0.704 | 0.711 | 0.273 | 0.100 |
|  |  | false positives per case | 0.53 | 0.62 | 4.21 | 13.15 |
| WAW-TACE | 17 | case tumour Dice (mean) | 0.662 | 0.650 | 0.640 | 0.638 |
|  |  | voxel Dice (pooled) | 0.829 | 0.837 | 0.838 | 0.822 |
|  |  | voxel recall (pooled) | 0.854 | 0.848 | 0.857 | 0.835 |
|  |  | voxel precision (pooled) | 0.805 | 0.826 | 0.820 | 0.808 |
|  |  | lesion recall = sensitivity | 0.833 | 0.833 | 0.833 | 0.833 |
|  |  | lesion precision | 0.541 | 0.571 | 0.345 | 0.113 |
|  |  | lesion F1 | 0.656 | 0.678 | 0.488 | 0.199 |
|  |  | false positives per case | 1.00 | 0.88 | 2.24 | 9.24 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 98 | Plain 3D U-Net, stratified split (small lesions in training) | 0.286 [0.21, 0.38] | 0.431 | 0.288 | 0.160 | 37 |
| **5-15 mm** | 98 | U-Net, lesion-balanced sampling (stratified split) | 0.316 [0.23, 0.41] | 0.437 | 0.319 | 0.174 | 40 |
| **5-15 mm** | 98 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.418 [0.33, 0.52] | 0.320 | 0.366 | 0.239 | 87 |
| **5-15 mm** | 98 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.520 [0.42, 0.62] | 0.229 | 0.428 | 0.306 | 172 |
| 0-5 mm (exploratory) | 53 | Plain 3D U-Net, stratified split (small lesions in training) | 0.057 [0.02, 0.15] | 0.136 | 0.030 | 0.010 | 19 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion-balanced sampling (stratified split) | 0.094 [0.04, 0.20] | 0.185 | 0.053 | 0.006 | 22 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.132 [0.07, 0.25] | 0.034 | 0.126 | 0.034 | 197 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.245 [0.15, 0.38] | 0.018 | 0.216 | 0.051 | 692 |
| 5-10 mm | 58 | Plain 3D U-Net, stratified split (small lesions in training) | 0.172 [0.10, 0.29] | 0.286 | 0.127 | 0.082 | 25 |
| 5-10 mm | 58 | U-Net, lesion-balanced sampling (stratified split) | 0.155 [0.08, 0.27] | 0.250 | 0.141 | 0.085 | 27 |
| 5-10 mm | 58 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.293 [0.19, 0.42] | 0.198 | 0.176 | 0.153 | 69 |
| 5-10 mm | 58 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.448 [0.33, 0.58] | 0.148 | 0.299 | 0.245 | 150 |
| 10-15 mm | 40 | Plain 3D U-Net, stratified split (small lesions in training) | 0.450 [0.31, 0.60] | 0.600 | 0.337 | 0.275 | 12 |
| 10-15 mm | 40 | U-Net, lesion-balanced sampling (stratified split) | 0.550 [0.40, 0.69] | 0.629 | 0.373 | 0.303 | 13 |
| 10-15 mm | 40 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.600 [0.45, 0.74] | 0.571 | 0.424 | 0.364 | 18 |
| 10-15 mm | 40 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.625 [0.47, 0.76] | 0.532 | 0.467 | 0.396 | 22 |
| > 15 mm | 154 | Plain 3D U-Net, stratified split (small lesions in training) | 0.890 [0.83, 0.93] | 0.913 | 0.867 | 0.680 | 13 |
| > 15 mm | 154 | U-Net, lesion-balanced sampling (stratified split) | 0.909 [0.85, 0.95] | 0.927 | 0.868 | 0.681 | 11 |
| > 15 mm | 154 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.929 [0.88, 0.96] | 0.929 | 0.868 | 0.684 | 11 |
| > 15 mm | 154 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.935 [0.88, 0.96] | 0.894 | 0.863 | 0.695 | 17 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 94 | Plain 3D U-Net, stratified split (small lesions in training) | 0.287 [0.21, 0.39] | 0.551 | 0.285 | 0.159 | 22 |
| **5-15 mm** | 94 | U-Net, lesion-balanced sampling (stratified split) | 0.319 [0.23, 0.42] | 0.556 | 0.317 | 0.174 | 24 |
| **5-15 mm** | 94 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.426 [0.33, 0.53] | 0.488 | 0.368 | 0.241 | 42 |
| **5-15 mm** | 94 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.532 [0.43, 0.63] | 0.410 | 0.434 | 0.312 | 72 |
| 0-5 mm (exploratory) | 53 | Plain 3D U-Net, stratified split (small lesions in training) | 0.057 [0.02, 0.15] | 0.231 | 0.030 | 0.010 | 10 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion-balanced sampling (stratified split) | 0.094 [0.04, 0.20] | 0.294 | 0.053 | 0.006 | 12 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.132 [0.07, 0.25] | 0.090 | 0.126 | 0.034 | 71 |
| 0-5 mm (exploratory) | 53 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.245 [0.15, 0.38] | 0.060 | 0.216 | 0.051 | 203 |
| 5-10 mm | 57 | Plain 3D U-Net, stratified split (small lesions in training) | 0.175 [0.10, 0.29] | 0.385 | 0.129 | 0.083 | 16 |
| 5-10 mm | 57 | U-Net, lesion-balanced sampling (stratified split) | 0.158 [0.09, 0.27] | 0.300 | 0.144 | 0.086 | 21 |
| 5-10 mm | 57 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.298 [0.20, 0.43] | 0.333 | 0.180 | 0.156 | 34 |
| 5-10 mm | 57 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.456 [0.33, 0.58] | 0.299 | 0.306 | 0.249 | 61 |
| 10-15 mm | 37 | Plain 3D U-Net, stratified split (small lesions in training) | 0.459 [0.31, 0.62] | 0.739 | 0.335 | 0.277 | 6 |
| 10-15 mm | 37 | U-Net, lesion-balanced sampling (stratified split) | 0.568 [0.41, 0.71] | 0.875 | 0.372 | 0.309 | 3 |
| 10-15 mm | 37 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.622 [0.46, 0.76] | 0.742 | 0.429 | 0.374 | 8 |
| 10-15 mm | 37 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.649 [0.49, 0.78] | 0.686 | 0.475 | 0.408 | 11 |
| > 15 mm | 106 | Plain 3D U-Net, stratified split (small lesions in training) | 0.877 [0.80, 0.93] | 0.979 | 0.872 | 0.655 | 2 |
| > 15 mm | 106 | U-Net, lesion-balanced sampling (stratified split) | 0.887 [0.81, 0.93] | 0.989 | 0.874 | 0.659 | 1 |
| > 15 mm | 106 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.915 [0.85, 0.95] | 0.990 | 0.869 | 0.665 | 1 |
| > 15 mm | 106 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.943 [0.88, 0.97] | 0.980 | 0.873 | 0.689 | 2 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 1 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 9 |
| **5-15 mm** | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 9 |
| **5-15 mm** | 1 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 34 |
| **5-15 mm** | 1 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 86 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 4 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | 0.000 | n/a | n/a | 103 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | 0.000 | n/a | n/a | 353 |
| 5-10 mm | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | 0.000 | n/a | n/a | 28 |
| 5-10 mm | 0 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | 0.000 | n/a | n/a | 78 |
| 10-15 mm | 1 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 1 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 6 |
| 10-15 mm | 1 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 8 |
| > 15 mm | 27 | Plain 3D U-Net, stratified split (small lesions in training) | 0.926 [0.77, 0.98] | 0.833 | 0.865 | 0.748 | 5 |
| > 15 mm | 27 | U-Net, lesion-balanced sampling (stratified split) | 1.000 [0.88, 1.00] | 0.844 | 0.869 | 0.776 | 5 |
| > 15 mm | 27 | U-Net, lesion sampling + synthetic tumours (stratified) | 1.000 [0.88, 1.00] | 0.818 | 0.871 | 0.773 | 6 |
| > 15 mm | 27 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.926 [0.77, 0.98] | 0.758 | 0.862 | 0.744 | 8 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 3 | Plain 3D U-Net, stratified split (small lesions in training) | 0.333 [0.06, 0.79] | 0.143 | 0.398 | 0.245 | 6 |
| **5-15 mm** | 3 | U-Net, lesion-balanced sampling (stratified split) | 0.333 [0.06, 0.79] | 0.125 | 0.416 | 0.231 | 7 |
| **5-15 mm** | 3 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.333 [0.06, 0.79] | 0.083 | 0.410 | 0.238 | 11 |
| **5-15 mm** | 3 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.333 [0.06, 0.79] | 0.067 | 0.412 | 0.242 | 14 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 5 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.000 | n/a | n/a | 3 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | 0.000 | n/a | n/a | 23 |
| 0-5 mm (exploratory) | 0 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | 0.000 | n/a | n/a | 136 |
| 5-10 mm | 1 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 1 | U-Net, lesion-balanced sampling (stratified split) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 2 |
| 5-10 mm | 1 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 7 |
| 5-10 mm | 1 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 11 |
| 10-15 mm | 2 | Plain 3D U-Net, stratified split (small lesions in training) | 0.500 [0.09, 0.91] | 0.500 | 0.438 | 0.368 | 1 |
| 10-15 mm | 2 | U-Net, lesion-balanced sampling (stratified split) | 0.500 [0.09, 0.91] | 0.167 | 0.458 | 0.347 | 5 |
| 10-15 mm | 2 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.500 [0.09, 0.91] | 0.200 | 0.450 | 0.357 | 4 |
| 10-15 mm | 2 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.500 [0.09, 0.91] | 0.250 | 0.453 | 0.363 | 3 |
| > 15 mm | 21 | Plain 3D U-Net, stratified split (small lesions in training) | 0.905 [0.71, 0.97] | 0.760 | 0.854 | 0.716 | 6 |
| > 15 mm | 21 | U-Net, lesion-balanced sampling (stratified split) | 0.905 [0.71, 0.97] | 0.792 | 0.849 | 0.671 | 5 |
| > 15 mm | 21 | U-Net, lesion sampling + synthetic tumours (stratified) | 0.905 [0.71, 0.97] | 0.826 | 0.857 | 0.669 | 4 |
| > 15 mm | 21 | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | 0.905 [0.71, 0.97] | 0.731 | 0.836 | 0.662 | 7 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.260 | 0.332 | 0.394 | 0.375 |
| 5-15 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | 0.278 | 0.357 | 0.415 |
| 5-15 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.464 |
| 5-10 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.121 | 0.182 | 0.234 | 0.231 |
| 5-10 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | 0.156 | 0.207 | 0.299 |
| 5-10 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.376 |
| 10-15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.463 | 0.550 | 0.625 | 0.584 |
| 10-15 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | 0.455 | 0.575 | 0.582 |
| 10-15 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.590 |
| > 15 mm | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.896 | 0.913 | 0.943 | 0.932 |
| > 15 mm | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | 0.838 | 0.912 | 0.908 |
| > 15 mm | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.898 |
| all sizes | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.548 | 0.584 | 0.623 | 0.612 |
| all sizes | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | 0.523 | 0.595 | 0.616 |
| all sizes | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.630 |
| 0-5 mm (exploratory) | U-Net, lesion-balanced sampling (stratified split) | n/a | 0.066 | 0.094 | 0.113 | 0.123 |
| 0-5 mm (exploratory) | U-Net, lesion sampling + synthetic tumours (stratified) | n/a | n/a | 0.058 | 0.113 | 0.142 |
| 0-5 mm (exploratory) | U-Net, lesion sampling + synthetic + CC-DiceCE (stratified) | n/a | n/a | n/a | n/a | 0.157 |

### Training

```
Plain 3D U-Net, stratified split (small lesions in training): {"epochs_logged": 300, "mean_epoch_seconds": 49.0, "final_val_patch_dice_liver_tumour": [0.9447, 0.8548], "best_val_patch_dice_tumour": 0.8928, "final_train_loss": 0.3325, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net, lesion-balanced sampling (stratified split): {"epochs_logged": 300, "mean_epoch_seconds": 49.1, "final_val_patch_dice_liver_tumour": [0.9582, 0.8351], "best_val_patch_dice_tumour": 0.8921, "final_train_loss": 0.3521, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net, lesion sampling + synthetic tumours (stratified): {"epochs_logged": 300, "mean_epoch_seconds": 50.1, "final_val_patch_dice_liver_tumour": [0.958, 0.8406], "best_val_patch_dice_tumour": 0.891, "final_train_loss": 0.3047, "synthetic_lesions_per_epoch_mean": 492.5}
U-Net, lesion sampling + synthetic + CC-DiceCE (stratified): {"epochs_logged": 300, "mean_epoch_seconds": 67.2, "final_val_patch_dice_liver_tumour": [0.9566, 0.8426], "best_val_patch_dice_tumour": 0.8866, "final_train_loss": 0.3984, "synthetic_lesions_per_epoch_mean": 492.5}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Artefacts

`results/2026-10-03_postprocessing_diagnostic_stratified/`: strat_lesion_config.json, strat_lesion_history.json, strat_lesion_jobs.json, strat_lesion_synth_cc_config.json, strat_lesion_synth_cc_history.json, strat_lesion_synth_cc_jobs.json, strat_lesion_synth_cc_test_evaluation_minsize100_size_binned_metrics.csv, strat_lesion_synth_cc_test_froc.csv, strat_lesion_synth_cc_test_froc_min33_froc_summary.json, strat_lesion_synth_cc_test_froc_summary.json, strat_lesion_synth_cc_test_per_lesion.csv, strat_lesion_synth_cc_test_size_binned_metrics.csv, strat_lesion_synth_cc_val_froc.csv, strat_lesion_synth_cc_val_froc_min33_froc_summary.json, strat_lesion_synth_cc_val_froc_summary.json, strat_lesion_synth_cc_val_per_lesion.csv, strat_lesion_synth_cc_val_size_binned_metrics.csv, strat_lesion_synth_config.json, strat_lesion_synth_history.json, strat_lesion_synth_jobs.json, strat_lesion_synth_test_evaluation_minsize100_size_binned_metrics.csv, strat_lesion_synth_test_froc.csv, strat_lesion_synth_test_froc_min33_froc_summary.json, strat_lesion_synth_test_froc_summary.json, strat_lesion_synth_test_per_lesion.csv, strat_lesion_synth_test_size_binned_metrics.csv, strat_lesion_synth_val_froc.csv, strat_lesion_synth_val_froc_min33_froc_summary.json, strat_lesion_synth_val_froc_summary.json, strat_lesion_synth_val_per_lesion.csv, strat_lesion_synth_val_size_binned_metrics.csv, strat_lesion_test_evaluation_minsize100_size_binned_metrics.csv, strat_lesion_test_froc.csv, strat_lesion_test_froc_min33_froc_summary.json, strat_lesion_test_froc_summary.json, strat_lesion_test_per_lesion.csv, strat_lesion_test_size_binned_metrics.csv, strat_lesion_val_froc.csv, strat_lesion_val_froc_min33_froc_summary.json, strat_lesion_val_froc_summary.json, strat_lesion_val_per_lesion.csv, strat_lesion_val_size_binned_metrics.csv, stratified_config.json, stratified_history.json, stratified_jobs.json, stratified_test_per_lesion.csv, stratified_test_size_binned_metrics.csv, stratified_val_per_lesion.csv, stratified_val_size_binned_metrics.csv

