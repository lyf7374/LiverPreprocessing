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
