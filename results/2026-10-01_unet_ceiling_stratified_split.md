# 2026-10-01: unet ceiling stratified split

**Purpose.** Ceiling experiment: how sensitive can the current plain 3D U-Net (1 mm grid, nnU-Net plans, 300 epochs) become on 5-15 mm lesions when real small lesions ARE in the training set? The answer separates the two candidate explanations of the low small-lesion sensitivity on the small-held-out split: missing training signal (data) versus resolution / architecture / annotation quality. Compared with the same network trained on the small-held-out split (reference columns; note that its test subset is different: 439 cases with 840 lesions of 5-15 mm versus 210 cases with 129 such lesions here, so only per-bin sensitivities are comparable, not case counts).

**Data.** unified_v2 (MCT-LTDiag 517, PLC-CECT 361, WAW-TACE 164 patients), four registered phases NC / AP / PVP / DP as int16 HU on a 1 mm isotropic liver-centred grid, plus the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT). Split: stratified seed 0 (train 727 / val 105 / test 210; lesions of every size in all subsets); no lesion <= 15 mm in training or validation; 17 PLC training cases with a phase below the registration gate were dropped; WAW-TACE 33 and 34 excluded.

**Setup.**

Same code, architecture (30,789,391 parameters), recipe and augmentation as the plain 3D U-Net baseline (300 epochs x 250 iterations, batch 2, patch 112 x 128 x 160, SGD 0.01, CE + Dice deep supervision, 33 % foreground-voxel sampling, mirroring at test time), only the split differs: make_splits.py protocol stratified (iterative stratification over dataset, diagnosis, lesion presence and size bins, seed 0), train 727 patients (708 after dropping the 19 with an unusable phase), val 105, test 210; the training set therefore contains about a quarter of all small-lesion patients. Selected through the new SPLIT_OVERRIDE variable of remote/environment.sh; run unet3d_run/stratified, jobs 29604263 (train, 4.1 h, 49 s per epoch) and 29604264 (predict + evaluate, 33 min).

**Compute.** QMUL Apocrita, partition andrena, one A100 40 GB per job, 8 CPUs, 88 GB RAM.

**Runs.** `unet3d` = Plain 3D U-Net 300 ep; `stratified` = Plain 3D U-Net, stratified split (small lesions in training)

**Findings.**

With small lesions in training the plain U-Net reaches, on the stratified test subset (210 cases): 5-15 mm sensitivity 0.403 [0.32, 0.49] (129 lesions), 5-10 mm 0.213 [0.13, 0.33] (61), 10-15 mm 0.574 [0.46, 0.68] (68), > 15 mm 0.937; overall lesion recall 0.647, lesion precision 0.639, 0.88 false positives per case, case tumour Dice 0.703. On the validation subset (105 cases, 98 lesions of 5-15 mm) the 5-15 mm sensitivity is 0.286 (5-10 mm 0.172, 10-15 mm 0.450). For reference, the same network trained without small lesions gives 0.226 on the 840 small-held-out test lesions (5-10 mm 0.061, 10-15 mm 0.441), the synthetic-tumour run 0.405 and CC-DiceCE 0.289. Interpretation: real small lesions in training roughly double the 5-15 mm sensitivity but the ceiling of this architecture and recipe is about 0.4 (0.2 for 5-10 mm), far below the 0.6-0.7 that would indicate a pure data problem, and the synthetic-tumour run already sits at that ceiling. The remaining gap must be addressed on the resolution / representation / sampling side rather than by more small-lesion labels alone: candidates are lesion-balanced foreground sampling (the current sampler picks foreground voxels, so large lesions dominate), a detection-style model (nnDetection), higher in-plane resolution within the liver, contrast-difference input channels and a single-phase control for registration error, plus a two-stage false-positive reducer at native resolution. Caveats: the 5-15 mm test count is small (129 lesions, CI width 0.17); the dataset mix of the stratified split differs from the small-held-out split (here MCT-LTDiag 104, PLC-CECT 73, WAW-TACE 33 test cases), and 96 % of the small lesions are MCT-LTDiag metastases whose native slice thickness is 5 mm.

## Test subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | Plain 3D U-Net 300 ep | Plain 3D U-Net, stratified split (small lesions in training) |
|---|---:|---|---:|---:|
| overall | 439 | case tumour Dice (mean) | 0.709 | 0.703 |
|  |  | voxel Dice (pooled) | 0.851 | 0.856 |
|  |  | voxel recall (pooled) | 0.827 | 0.854 |
|  |  | voxel precision (pooled) | 0.877 | 0.858 |
|  |  | lesion recall = sensitivity | 0.413 | 0.647 |
|  |  | lesion precision | 0.797 | 0.639 |
|  |  | lesion F1 | 0.544 | 0.643 |
|  |  | false positives per case | 0.50 | 0.88 |
| MCT-LTDiag | 354 | case tumour Dice (mean) | 0.730 | 0.756 |
|  |  | voxel Dice (pooled) | 0.856 | 0.895 |
|  |  | voxel recall (pooled) | 0.827 | 0.899 |
|  |  | voxel precision (pooled) | 0.887 | 0.890 |
|  |  | lesion recall = sensitivity | 0.396 | 0.573 |
|  |  | lesion precision | 0.821 | 0.749 |
|  |  | lesion F1 | 0.534 | 0.649 |
|  |  | false positives per case | 0.47 | 0.71 |
| PLC-CECT | 48 | case tumour Dice (mean) | 0.537 | 0.602 |
|  |  | voxel Dice (pooled) | 0.843 | 0.816 |
|  |  | voxel recall (pooled) | 0.835 | 0.779 |
|  |  | voxel precision (pooled) | 0.852 | 0.856 |
|  |  | lesion recall = sensitivity | 0.659 | 0.937 |
|  |  | lesion precision | 0.635 | 0.472 |
|  |  | lesion F1 | 0.647 | 0.628 |
|  |  | false positives per case | 0.65 | 0.96 |
| WAW-TACE | 37 | case tumour Dice (mean) | 0.717 | 0.722 |
|  |  | voxel Dice (pooled) | 0.800 | 0.765 |
|  |  | voxel recall (pooled) | 0.810 | 0.850 |
|  |  | voxel precision (pooled) | 0.790 | 0.695 |
|  |  | lesion recall = sensitivity | 0.587 | 0.849 |
|  |  | lesion precision | 0.667 | 0.506 |
|  |  | lesion F1 | 0.624 | 0.634 |
|  |  | false positives per case | 0.59 | 1.26 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 840 | Plain 3D U-Net 300 ep | 0.226 [0.20, 0.26] | 0.638 | 0.264 | 0.116 | 108 |
| **5-15 mm** | 129 | Plain 3D U-Net, stratified split (small lesions in training) | 0.403 [0.32, 0.49] | 0.354 | 0.346 | 0.205 | 95 |
| 0-5 mm (exploratory) | 509 | Plain 3D U-Net 300 ep | 0.065 [0.05, 0.09] | 0.337 | 0.051 | 0.001 | 65 |
| 0-5 mm (exploratory) | 91 | Plain 3D U-Net, stratified split (small lesions in training) | 0.077 [0.04, 0.15] | 0.115 | 0.051 | 0.002 | 54 |
| 5-10 mm | 475 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.312 | 0.058 | 0.019 | 64 |
| 5-10 mm | 61 | Plain 3D U-Net, stratified split (small lesions in training) | 0.213 [0.13, 0.33] | 0.167 | 0.137 | 0.060 | 65 |
| 10-15 mm | 365 | Plain 3D U-Net 300 ep | 0.441 [0.39, 0.49] | 0.785 | 0.322 | 0.242 | 44 |
| 10-15 mm | 68 | Plain 3D U-Net, stratified split (small lesions in training) | 0.574 [0.46, 0.68] | 0.565 | 0.390 | 0.335 | 30 |
| > 15 mm | 742 | Plain 3D U-Net 300 ep | 0.863 [0.84, 0.89] | 0.932 | 0.830 | 0.640 | 47 |
| > 15 mm | 287 | Plain 3D U-Net, stratified split (small lesions in training) | 0.937 [0.90, 0.96] | 0.882 | 0.855 | 0.709 | 36 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 781 | Plain 3D U-Net 300 ep | 0.224 [0.20, 0.25] | 0.670 | 0.267 | 0.115 | 86 |
| **5-15 mm** | 118 | Plain 3D U-Net, stratified split (small lesions in training) | 0.407 [0.32, 0.50] | 0.516 | 0.341 | 0.204 | 45 |
| 0-5 mm (exploratory) | 508 | Plain 3D U-Net 300 ep | 0.063 [0.04, 0.09] | 0.421 | 0.051 | 0.001 | 44 |
| 0-5 mm (exploratory) | 91 | Plain 3D U-Net, stratified split (small lesions in training) | 0.077 [0.04, 0.15] | 0.280 | 0.051 | 0.002 | 18 |
| 5-10 mm | 458 | Plain 3D U-Net 300 ep | 0.061 [0.04, 0.09] | 0.346 | 0.059 | 0.020 | 53 |
| 5-10 mm | 57 | Plain 3D U-Net, stratified split (small lesions in training) | 0.228 [0.14, 0.35] | 0.283 | 0.148 | 0.064 | 33 |
| 10-15 mm | 323 | Plain 3D U-Net 300 ep | 0.455 [0.40, 0.51] | 0.817 | 0.330 | 0.249 | 33 |
| 10-15 mm | 61 | Plain 3D U-Net, stratified split (small lesions in training) | 0.574 [0.45, 0.69] | 0.745 | 0.384 | 0.334 | 12 |
| > 15 mm | 645 | Plain 3D U-Net 300 ep | 0.865 [0.84, 0.89] | 0.938 | 0.830 | 0.643 | 37 |
| > 15 mm | 182 | Plain 3D U-Net, stratified split (small lesions in training) | 0.929 [0.88, 0.96] | 0.934 | 0.901 | 0.700 | 12 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 32 | Plain 3D U-Net 300 ep | 0.344 [0.20, 0.52] | 0.458 | 0.276 | 0.166 | 13 |
| **5-15 mm** | 5 | Plain 3D U-Net, stratified split (small lesions in training) | 0.600 [0.23, 0.88] | 0.088 | 0.450 | 0.330 | 31 |
| 0-5 mm (exploratory) | 1 | Plain 3D U-Net 300 ep | 1.000 [0.21, 1.00] | 0.067 | 1.000 | 0.000 | 14 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 23 |
| 5-10 mm | 9 | Plain 3D U-Net 300 ep | 0.111 [0.02, 0.44] | 0.143 | 0.087 | 0.000 | 6 |
| 5-10 mm | 2 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 21 |
| 10-15 mm | 23 | Plain 3D U-Net 300 ep | 0.435 [0.26, 0.63] | 0.588 | 0.306 | 0.232 | 7 |
| 10-15 mm | 3 | Plain 3D U-Net, stratified split (small lesions in training) | 1.000 [0.44, 1.00] | 0.231 | 0.539 | 0.550 | 10 |
| > 15 mm | 49 | Plain 3D U-Net 300 ep | 0.857 [0.73, 0.93] | 0.913 | 0.836 | 0.566 | 4 |
| > 15 mm | 58 | Plain 3D U-Net, stratified split (small lesions in training) | 0.966 [0.88, 0.99] | 0.824 | 0.779 | 0.708 | 12 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 27 | Plain 3D U-Net 300 ep | 0.148 [0.06, 0.32] | 0.308 | 0.194 | 0.094 | 9 |
| **5-15 mm** | 6 | Plain 3D U-Net, stratified split (small lesions in training) | 0.167 [0.03, 0.56] | 0.050 | 0.333 | 0.130 | 19 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 7 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 13 |
| 5-10 mm | 8 | Plain 3D U-Net 300 ep | 0.000 [0.00, 0.32] | 0.000 | 0.000 | 0.000 | 5 |
| 5-10 mm | 2 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.66] | 0.000 | 0.000 | 0.000 | 11 |
| 10-15 mm | 19 | Plain 3D U-Net 300 ep | 0.211 [0.09, 0.43] | 0.500 | 0.219 | 0.134 | 4 |
| 10-15 mm | 4 | Plain 3D U-Net, stratified split (small lesions in training) | 0.250 [0.05, 0.70] | 0.111 | 0.369 | 0.195 | 8 |
| > 15 mm | 48 | Plain 3D U-Net 300 ep | 0.833 [0.70, 0.91] | 0.870 | 0.813 | 0.675 | 6 |
| > 15 mm | 47 | Plain 3D U-Net, stratified split (small lesions in training) | 0.936 [0.83, 0.98] | 0.786 | 0.851 | 0.742 | 12 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | Plain 3D U-Net 300 ep | 0.178 | 0.235 | 0.288 | 0.336 | 0.283 |
| 5-10 mm | Plain 3D U-Net 300 ep | 0.045 | 0.068 | 0.103 | 0.151 | 0.112 |
| 10-15 mm | Plain 3D U-Net 300 ep | 0.352 | 0.452 | 0.529 | 0.575 | 0.506 |
| > 15 mm | Plain 3D U-Net 300 ep | 0.820 | 0.869 | 0.899 | 0.925 | 0.891 |
| all sizes | Plain 3D U-Net 300 ep | 0.371 | 0.420 | 0.459 | 0.492 | 0.453 |
| 0-5 mm (exploratory) | Plain 3D U-Net 300 ep | 0.033 | 0.070 | 0.098 | 0.121 | 0.093 |

### Training

```
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
Plain 3D U-Net, stratified split (small lesions in training): {"epochs_logged": 300, "mean_epoch_seconds": 49.0, "final_val_patch_dice_liver_tumour": [0.9447, 0.8548], "best_val_patch_dice_tumour": 0.8928, "final_train_loss": 0.3325, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Val subset


Split: small-held-out seed 0 (train 534 / val 67 / test 439; no lesion <= 15 mm in training or validation). Lesion = 26-connected component, size = equivalent spherical diameter on the 1 mm grid. A lesion is detected when >= 10 % of its voxels are covered by the prediction; a predicted component touching no lesion is a false positive. Sensitivity is lesion-level recall (95 % Wilson interval); lesion precision = detected / (detected + false-positive components of that size); voxel recall = covered lesion voxels / lesion voxels; lesion Dice = mean per-lesion Dice (0 for a miss); case Dice = mean per-case tumour Dice; pooled voxel Dice / recall / precision are computed over all voxels of the scope.

### Summary per scope (all lesion sizes)

| Scope | Cases | Metric | Plain 3D U-Net 300 ep | Plain 3D U-Net, stratified split (small lesions in training) |
|---|---:|---|---:|---:|
| overall | 67 | case tumour Dice (mean) | 0.702 | 0.693 |
|  |  | voxel Dice (pooled) | 0.784 | 0.845 |
|  |  | voxel recall (pooled) | 0.753 | 0.864 |
|  |  | voxel precision (pooled) | 0.818 | 0.826 |
|  |  | lesion recall = sensitivity | 0.959 | 0.551 |
|  |  | lesion precision | 0.617 | 0.709 |
|  |  | lesion F1 | 0.751 | 0.620 |
|  |  | false positives per case | 0.66 | 0.66 |
| MCT-LTDiag | 19 | case tumour Dice (mean) | 0.788 | 0.763 |
|  |  | voxel Dice (pooled) | 0.859 | 0.869 |
|  |  | voxel recall (pooled) | 0.851 | 0.867 |
|  |  | voxel precision (pooled) | 0.868 | 0.871 |
|  |  | lesion recall = sensitivity | 0.960 | 0.486 |
|  |  | lesion precision | 0.774 | 0.783 |
|  |  | lesion F1 | 0.857 | 0.600 |
|  |  | false positives per case | 0.37 | 0.63 |
| PLC-CECT | 34 | case tumour Dice (mean) | 0.614 | 0.589 |
|  |  | voxel Dice (pooled) | 0.751 | 0.821 |
|  |  | voxel recall (pooled) | 0.734 | 0.865 |
|  |  | voxel precision (pooled) | 0.770 | 0.782 |
|  |  | lesion recall = sensitivity | 0.926 | 0.893 |
|  |  | lesion precision | 0.446 | 0.581 |
|  |  | lesion F1 | 0.602 | 0.704 |
|  |  | false positives per case | 0.91 | 0.53 |
| WAW-TACE | 14 | case tumour Dice (mean) | 0.757 | 0.662 |
|  |  | voxel Dice (pooled) | 0.817 | 0.829 |
|  |  | voxel recall (pooled) | 0.729 | 0.854 |
|  |  | voxel precision (pooled) | 0.929 | 0.805 |
|  |  | lesion recall = sensitivity | 1.000 | 0.833 |
|  |  | lesion precision | 0.786 | 0.541 |
|  |  | lesion F1 | 0.880 | 0.656 |
|  |  | false positives per case | 0.43 | 1.00 |

### Per size bin

Primary target bin: 5-15 mm (union of 5-10 and 10-15 mm). The 0-5 mm bin is exploratory: about half of its ground-truth components are <= 10 voxels and all lie in cases with larger lesions (annotation fragments).

#### overall

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 98 | Plain 3D U-Net, stratified split (small lesions in training) | 0.286 [0.21, 0.38] | 0.431 | 0.288 | 0.160 | 37 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 18 |
| 0-5 mm (exploratory) | 53 | Plain 3D U-Net, stratified split (small lesions in training) | 0.057 [0.02, 0.15] | 0.136 | 0.030 | 0.010 | 19 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 10 |
| 5-10 mm | 58 | Plain 3D U-Net, stratified split (small lesions in training) | 0.172 [0.10, 0.29] | 0.286 | 0.127 | 0.082 | 25 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 9 |
| 10-15 mm | 40 | Plain 3D U-Net, stratified split (small lesions in training) | 0.450 [0.31, 0.60] | 0.600 | 0.337 | 0.275 | 12 |
| > 15 mm | 74 | Plain 3D U-Net 300 ep | 0.959 [0.89, 0.99] | 0.910 | 0.753 | 0.726 | 7 |
| > 15 mm | 154 | Plain 3D U-Net, stratified split (small lesions in training) | 0.890 [0.83, 0.93] | 0.913 | 0.867 | 0.680 | 13 |

#### MCT-LTDiag

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 94 | Plain 3D U-Net, stratified split (small lesions in training) | 0.287 [0.21, 0.39] | 0.551 | 0.285 | 0.159 | 22 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 2 |
| 0-5 mm (exploratory) | 53 | Plain 3D U-Net, stratified split (small lesions in training) | 0.057 [0.02, 0.15] | 0.231 | 0.030 | 0.010 | 10 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 5-10 mm | 57 | Plain 3D U-Net, stratified split (small lesions in training) | 0.175 [0.10, 0.29] | 0.385 | 0.129 | 0.083 | 16 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 37 | Plain 3D U-Net, stratified split (small lesions in training) | 0.459 [0.31, 0.62] | 0.739 | 0.335 | 0.277 | 6 |
| > 15 mm | 25 | Plain 3D U-Net 300 ep | 0.960 [0.80, 0.99] | 1.000 | 0.851 | 0.776 | 0 |
| > 15 mm | 106 | Plain 3D U-Net, stratified split (small lesions in training) | 0.877 [0.80, 0.93] | 0.979 | 0.872 | 0.655 | 2 |

#### PLC-CECT

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 1 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 9 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 15 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 10-15 mm | 1 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| > 15 mm | 27 | Plain 3D U-Net 300 ep | 0.926 [0.77, 0.98] | 0.781 | 0.734 | 0.681 | 7 |
| > 15 mm | 27 | Plain 3D U-Net, stratified split (small lesions in training) | 0.926 [0.77, 0.98] | 0.833 | 0.865 | 0.748 | 5 |

#### WAW-TACE

| Size bin | Lesions | Run | Sensitivity [95 % CI] | Lesion precision | Voxel recall | Lesion Dice | FP components |
|---|---:|---|---|---:|---:|---:|---:|
| **5-15 mm** | 3 | Plain 3D U-Net, stratified split (small lesions in training) | 0.333 [0.06, 0.79] | 0.143 | 0.398 | 0.245 | 6 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 0-5 mm (exploratory) | 0 | Plain 3D U-Net, stratified split (small lesions in training) | n/a | 0.000 | n/a | n/a | 5 |
| 5-10 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 4 |
| 5-10 mm | 1 | Plain 3D U-Net, stratified split (small lesions in training) | 0.000 [0.00, 0.79] | 0.000 | 0.000 | 0.000 | 5 |
| 10-15 mm | 0 | Plain 3D U-Net 300 ep | n/a | 0.000 | n/a | n/a | 1 |
| 10-15 mm | 2 | Plain 3D U-Net, stratified split (small lesions in training) | 0.500 [0.09, 0.91] | 0.500 | 0.438 | 0.368 | 1 |
| > 15 mm | 22 | Plain 3D U-Net 300 ep | 1.000 [0.85, 1.00] | 1.000 | 0.729 | 0.724 | 0 |
| > 15 mm | 21 | Plain 3D U-Net, stratified split (small lesions in training) | 0.905 [0.71, 0.97] | 0.760 | 0.854 | 0.716 | 6 |

### FROC: sensitivity at fixed false positives per case (overall, from the probability maps)

| Size bin | Run | @0.25 FP/case | @0.5 FP/case | @1.0 FP/case | @2.0 FP/case | mean over measured rates |
|---|---|---:|---:|---:|---:|---:|
| 5-15 mm | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |
| 5-10 mm | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |
| 10-15 mm | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |
| > 15 mm | Plain 3D U-Net 300 ep | 0.936 | 0.959 | 0.973 | 0.973 | 0.964 |
| all sizes | Plain 3D U-Net 300 ep | 0.936 | 0.959 | 0.973 | 0.973 | 0.964 |
| 0-5 mm (exploratory) | Plain 3D U-Net 300 ep | n/a | n/a | n/a | n/a | n/a |

### Training

```
Plain 3D U-Net 300 ep: {"epochs_logged": 300, "mean_epoch_seconds": 113.0, "final_val_patch_dice_liver_tumour": [0.9585, 0.8506], "best_val_patch_dice_tumour": 0.8773, "final_train_loss": 0.33}
Plain 3D U-Net, stratified split (small lesions in training): {"epochs_logged": 300, "mean_epoch_seconds": 49.0, "final_val_patch_dice_liver_tumour": [0.9447, 0.8548], "best_val_patch_dice_tumour": 0.8928, "final_train_loss": 0.3325, "synthetic_lesions_per_epoch_mean": 0.0}
U-Net architecture (from nnUNetPlans.json): patch [112, 128, 160], batch 2, features [32, 64, 128, 256, 320, 320], batch_dice False
```

## Artefacts

`results/2026-10-01_unet_ceiling_stratified_split/`: stratified_config.json, stratified_history.json, stratified_jobs.json, stratified_test_per_lesion.csv, stratified_test_size_binned_metrics.csv, stratified_val_per_lesion.csv, stratified_val_size_binned_metrics.csv, unet3d_config.json, unet3d_history.json, unet3d_test_froc.csv, unet3d_test_froc_summary.json, unet3d_test_per_lesion.csv, unet3d_test_size_binned_metrics.csv, unet3d_val_froc.csv, unet3d_val_froc_summary.json, unet3d_val_per_lesion.csv, unet3d_val_size_binned_metrics.csv

