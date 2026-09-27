# Experiment records

One Markdown file per experiment, written by `experiments/baselines_v2/tools/record_experiment.py` from the fetched cluster results. Metrics come from `scripts/evaluate_by_lesion_size.py` (lesion-level sensitivity = recall, precision, F1; voxel-level Dice / recall / precision; per size bin 0-5, 5-10, 10-15, > 15 mm ESD; per dataset).

| Date | Experiment | Split | Test cases | Case Dice | Lesion recall | Lesion precision | FP / case | Record |
|---|---|---|---:|---|---|---|---|---|
| 2026-09-27 | baselines_small_held_out (nnU-Net 300 ep) | small-held-out seed 0 | 439 | 0.713 | 0.413 | 0.786 | 0.54 | [2026-09-27_baselines_small_held_out.md](2026-09-27_baselines_small_held_out.md) |
| 2026-09-27 | baselines_small_held_out (Plain 3D U-Net 300 ep) | small-held-out seed 0 | 439 | 0.709 | 0.413 | 0.797 | 0.50 | [2026-09-27_baselines_small_held_out.md](2026-09-27_baselines_small_held_out.md) |
