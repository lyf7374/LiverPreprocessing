# Experiment records

One Markdown file per experiment, written by `experiments/baselines_v2/tools/record_experiment.py` from the fetched cluster results. Metrics come from `scripts/evaluate_by_lesion_size.py` (lesion-level sensitivity = recall, precision, F1; voxel-level Dice / recall / precision; per size bin 0-5, 5-10, 10-15, > 15 mm ESD; per dataset).

| Date | Experiment | Split | Test cases | Case Dice | Lesion recall | Lesion precision | FP / case | Record |
|---|---|---|---:|---|---|---|---|---|
| 2026-09-27 | baselines_small_held_out (nnU-Net 300 ep) | small-held-out seed 0 | 439 | 0.713 | 0.413 | 0.786 | 0.54 | [2026-09-27_baselines_small_held_out.md](2026-09-27_baselines_small_held_out.md) |
| 2026-09-27 | baselines_small_held_out (Plain 3D U-Net 300 ep) | small-held-out seed 0 | 439 | 0.709 | 0.413 | 0.797 | 0.50 | [2026-09-27_baselines_small_held_out.md](2026-09-27_baselines_small_held_out.md) |
| 2026-09-27 | unet_synthetic_small_tumours (U-Net + synthetic small tumours) | small-held-out seed 0 | 439 | 0.704 | 0.508 | 0.583 | 1.73 | [2026-09-27_unet_synthetic_small_tumours.md](2026-09-27_unet_synthetic_small_tumours.md) |
| 2026-09-28 | unet_cc_dicece_loss (U-Net + CC-DiceCE (w = 0.5)) | small-held-out seed 0 | 439 | 0.703 | 0.455 | 0.774 | 0.63 | [2026-09-28_unet_cc_dicece_loss.md](2026-09-28_unet_cc_dicece_loss.md) |
| 2026-09-28 | unet_phase_disentanglement (U-Net + phase disentanglement (DisC-Diff style)) | small-held-out seed 0 | 439 | 0.709 | 0.406 | 0.787 | 0.52 | [2026-09-28_unet_phase_disentanglement.md](2026-09-28_unet_phase_disentanglement.md) |
| 2026-09-28 | plan_stage1_segmentation_branch (PLAN stage 1 (seg branch) 300 ep) | small-held-out seed 0 | 439 | 0.716 | 0.455 | 0.797 | 0.55 | [2026-09-28_plan_stage1_segmentation_branch.md](2026-09-28_plan_stage1_segmentation_branch.md) |
| 2026-09-29 | plan_stage2_full_model (PLAN stage 2 (seg + det + cls) 300 ep) | small-held-out seed 0 | 439 | 0.721 | 0.440 | 0.862 | 0.34 | [2026-09-29_plan_stage2_full_model.md](2026-09-29_plan_stage2_full_model.md) |
| 2026-10-01 | unet_ceiling_stratified_split (Plain 3D U-Net, stratified split (small lesions in training)) | stratified seed 0 | 210 | 0.703 | 0.647 | 0.639 | 0.88 | [2026-10-01_unet_ceiling_stratified_split.md](2026-10-01_unet_ceiling_stratified_split.md) |
| 2026-10-01 | unet_lesion_sampling_stratified (U-Net, lesion-balanced sampling (stratified split)) | stratified seed 0 | 210 | 0.696 | 0.663 | 0.649 | 0.87 | [2026-10-01_unet_lesion_sampling_stratified.md](2026-10-01_unet_lesion_sampling_stratified.md) |
| 2026-10-01 | unet_lesion_sampling_stratified (U-Net, lesion sampling + synthetic tumours (stratified)) | stratified seed 0 | 210 | 0.704 | 0.696 | 0.428 | 2.25 | [2026-10-01_unet_lesion_sampling_stratified.md](2026-10-01_unet_lesion_sampling_stratified.md) |
| 2026-10-01 | unet_lesion_sampling_stratified (U-Net, lesion sampling + synthetic + CC-DiceCE (stratified)) | stratified seed 0 | 210 | 0.693 | 0.742 | 0.194 | 7.43 | [2026-10-01_unet_lesion_sampling_stratified.md](2026-10-01_unet_lesion_sampling_stratified.md) |
| 2026-10-04 | plan_stratified_stage1 (PLAN stage 1 (seg branch), lesion sampling, stratified) | stratified seed 0 | 210 | 0.728 | 0.692 | 0.643 | 0.93 | [2026-10-04_plan_stratified_stage1.md](2026-10-04_plan_stratified_stage1.md) |
| 2026-10-05 | plan_stratified_stage2 (PLAN stage 2 (seg + det + cls), stratified) | stratified seed 0 | 210 | 0.740 | 0.671 | 0.719 | 0.63 | [2026-10-05_plan_stratified_stage2.md](2026-10-05_plan_stratified_stage2.md) |
| 2026-10-05 | plan_stratified_stage2 (PLAN stage 2 modified (finer lesion-branch levels, 100 queries), stratified) | stratified seed 0 | 210 | 0.743 | 0.673 | 0.713 | 0.65 | [2026-10-05_plan_stratified_stage2.md](2026-10-05_plan_stratified_stage2.md) |
