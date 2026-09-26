# LiverPreprocessing

Reproducible conversion of three public multiphase liver CT releases, **MCT-LTDiag**, **PLC-CECT** and **WAW-TACE**, into one representation that a single segmentation / detection network (U-Net style) can consume without any dataset-specific handling:

- four contrast phases per patient, channel order **NC, AP, PVP, DP**, registered to PVP;
- **int16 Hounsfield units**, no clipping and no normalisation in preprocessing;
- one **1 x 1 x 1 mm**, LPS-oriented (x -> left, y -> posterior, z -> superior) grid per patient, cropped to the liver plus a 20 mm margin;
- a binary liver mask and a binary tumour mask on the same grid;
- patient-level diagnosis labels, connected-lesion size statistics, per-phase registration quality (`usable_*` flags) and geometric sanity flags in a manifest.

Any subset of the three datasets can be processed, and each release folder can be given by its own path. The pipeline is deterministic given the raw releases; the reference outputs we obtained (per-volume PLC geometry, the full manifest, summary counts) are in `reference/` so that a rerun can be checked line by line.

Pipeline version: `unified_v2_hu_int16_1mm_iso_gated_registration_2026-09-24`.

## Why a pipeline is needed at all

The three releases are not on one physical scale, and the PLC-CECT headers are wrong in three ways. Everything below was established by reading the raw files and cross-checking anatomy; details are in the stage descriptions.

| Release | Geometry in the header | Intensity | What the pipeline does |
|---|---|---|---|
| MCT-LTDiag | correct (about 0.7-0.8 x 0.7-0.8 x 5 mm, four phases already on one grid) | HU | resample once to 1 mm |
| WAW-TACE | correct (0.54-0.98 mm in-plane, 0.44-7.5 mm slices), phases on their own grids | HU | register to PVP, resample once to 1 mm |
| PLC-CECT | **all 1,444 CT files claim 1 x 1 x 1 mm, origin 0, uint8**; the true in-plane spacing is 400 mm / 512 = 0.78125 mm; the slice spacing differs per patient *and per phase* (0.3-8 mm); **1,442 of 1,444 volumes are stored superior-first although the header says k -> superior** | 8-bit window of [-200, 200] HU | recover slice order and spacing per volume from anatomy (stage 0), map back to HU, register, resample once to 1 mm |

With the released headers a network sees PLC livers 22 % too small in-plane, compressed 1-5x along z by a different factor in every phase, upside down, and with an intensity scale that differs from the other two datasets. None of that can be fixed after training.

## Requirements

- Python 3.11 (Windows and Linux; developed on Windows 11)
- A CUDA GPU for stage 0 (TotalSegmentator 3 mm model, about 1.5 s per volume on an RTX 5000 Ada; CPU works but is slow). Stages 1-2 are CPU only.
- About 4 GB RAM per conversion worker (8 workers were used on a 32-core, 128 GB machine)
- Disk: the processed dataset is 48 GB for all three releases

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install TotalSegmentator      # stage 0 only; pulls nnunetv2 and torch
python -c "import torch; print(torch.cuda.is_available())"   # must print True for a GPU run
```

The TotalSegmentator 3 mm weights (~135 MB) are fetched automatically into `~/.totalsegmentator/nnunet/results` the first time stage 0 runs. For an offline machine fetch them beforehand with `python scripts/estimate_plc_geometry.py download-weights`, or copy that folder from another machine and point `TOTALSEG_WEIGHTS_PATH` at it.

If `pip` installs a CPU-only `torch`, install the CUDA build that matches your driver from <https://pytorch.org> (we used `torch 2.11.0+cu128` with `torchvision 0.26.0+cu128`). `requirements.txt` gives minimum versions; `requirements-lock.txt` is the exact `pip freeze` of the environment that produced the reference outputs (TotalSegmentator 2.18.0, nnunetv2 2.8.1, SimpleITK 2.5.6, NumPy 2.4.2, nibabel 5.3.3, SciPy 1.17.1, torch 2.11.0+cu128). Use the lock file when the goal is to reproduce the reference numbers exactly. Stage 0 uses TotalSegmentator task 297 (`Dataset297_TotalSegmentator_total_3mm_1559subj`, trainer `nnUNetTrainer_4000epochs_NoMirroring`, fold 0) as downloaded by TotalSegmentator 2.18.0; the weight store can be relocated with the environment variable `TOTALSEG_WEIGHTS_PATH`.

## Raw data layout

Obtain the three releases from their official sources (MCT-LTDiag: per-patient TAR archives plus `meta_info_patient.tab`; PLC-CECT: the released ZIP parts plus `patient_data.csv`, see the data paper <https://doi.org/10.1038/s41597-025-05125-2> and <https://github.com/ljwa2323/PLC_CECT>; WAW-TACE: the Hugging Face image and organ-mask folders plus `ct_hcc_metadata.csv` and `tumor_masks.zip`). Place each release in its own folder; the folders can live anywhere:

```text
<any path>/MCT-LTDiag/
    meta_info_patient.tab
    230218a1.tar                     one TAR per patient (517); each holds NIFTI/{nc,art,pvp,delay}.nii.gz,
    ...                              mask_pvp.nii.gz and liver_mask_pvp.nii.gz
<any path>/PLC-CECT/
    patient_data.csv                 patient_id, phase (P/C1/C2/C3), cancer_type, ct_path, mask_path, liver_mask_path
    raw-*.zip                        the released ZIP parts (1,444 CT, 1,444 liver masks, tumour masks)
<any path>/WAW-TACE/
    ct_hcc_metadata.csv              patient_id, ct_phase (0-3), ct_file_name, tumor_count, slice_thickness
    tumor_masks.zip                  <patient>_<phase>_<lesion>_tumor_seg.nrrd
    huggingface/images/<file>.nii.gz
    huggingface/organ_masks/<file>.nii.gz   TotalSegmentator-style organ labels; label 5 = liver
```

When the three folders share one parent and carry these names, `--raw-root <parent>` is enough. Otherwise give `--mct-root`, `--plc-root`, `--waw-root` explicitly; an explicit path always wins.

Notes for a fresh download:

- The PLC-CECT ZIP parts are indexed by member file name, so it does not matter how many parts the download was split into or what they are called, as long as every `ct_path`, `mask_path` and `liver_mask_path` of `patient_data.csv` exists in one of them.
- The WAW-TACE cohort is derived automatically (four phases present, tumour masks in exactly one phase). The reference release yields 164 patients; if your copy of the release yields another number the runner stops, and `--expected-waw-cases 0` disables that check.
- The reference outputs were produced from the releases as downloaded in September 2026. A later revision of a release can change counts.

## Run

Always run the runner with the interpreter of the environment above (stage 0 imports TotalSegmentator). Print the resolved release folders, the WAW cohort size and the exact stage commands without writing anything:

```bash
python scripts/run_unified_preprocessing.py \
    --raw-root /data/raw \
    --output-root /data/processed/unified \
    --workers 8 --dry-run
```

Full conversion of the three datasets (remove `--dry-run`). The same command resumes an interrupted run: measured PLC volumes and cases already at the current pipeline version are skipped. Complete cases produced by another pipeline version are never reused silently: the run stops and asks for `--replace-datasets` (or another `--output-root`).

```bash
python scripts/run_unified_preprocessing.py \
    --raw-root /data/raw \
    --output-root /data/processed/unified \
    --workers 8
```

A single dataset, with its folder given explicitly (stage 0 runs only when PLC-CECT is selected; the WAW cohort is only derived when WAW-TACE is selected):

```bash
python scripts/run_unified_preprocessing.py \
    --datasets MCT-LTDiag \
    --mct-root /somewhere/else/MCT-LTDiag \
    --output-root /data/processed/unified \
    --workers 8

python scripts/run_unified_preprocessing.py \
    --datasets PLC-CECT WAW-TACE \
    --plc-root /disk1/PLC-CECT --waw-root /disk2/WAW-TACE \
    --output-root /data/processed/unified \
    --workers 8
```

Rebuild cases produced by an older pipeline version in place. Every case (new or replaced) is built in `.staging/` and enters `cases/` with a single directory rename after all its files and its `case.json` are written, so `cases/<case_id>/` is only ever a complete old version, a complete new version, or absent; an interrupted run leaves at most a leftover folder under `.staging/`, which is never read:

```bash
python scripts/run_unified_preprocessing.py \
    --raw-root /data/raw --output-root /data/processed/unified \
    --workers 8 --replace-datasets PLC-CECT MCT-LTDiag WAW-TACE
```

Windows PowerShell uses the same options (`.\.venv\Scripts\python.exe scripts\run_unified_preprocessing.py --raw-root D:\raw ...`).

### Stages and run time

| Stage | Script | Input | Output | Time (reference run) |
|---|---|---|---|---|
| 0 | `estimate_plc_geometry.py all` | PLC-CECT release | `manifests/plc_geometry.csv`, `manifests/plc_geometry/` | 71 min for 1,444 volumes, one GPU |
| 1 | `preprocess_multiphase_ct.py` | releases + `plc_geometry.csv` | `cases/`, `dataset_manifest.csv`, `summary.json`, `failures.json` | about 5 h for 1,042 cases with 8 workers (MCT-LTDiag alone about 40 min; the registered datasets dominate) |
| 2 | `prepare_hierarchical_labels.py` | `cases/` | labels in `case.json` and the manifest, `lesion_size_statistics.csv`, `lesion_size_summary.csv`, `label_schema.json` | 3 min |

The runner stops at the first stage that fails. Every stage can also be started on its own with the options shown by `--help`.

Skipping stage 0: to reuse our PLC geometry instead of re-estimating it (no GPU needed), copy `reference/plc_geometry.csv` to `<output-root>/manifests/plc_geometry.csv` and `reference/liver_fallback/` to `<output-root>/manifests/plc_geometry/liver_fallback/`, then add `--reuse-plc-geometry` to the runner command. Stage 0 is then omitted and stages 1-2 run as usual:

```bash
mkdir -p /data/processed/unified/manifests/plc_geometry
cp reference/plc_geometry.csv /data/processed/unified/manifests/
cp -r reference/liver_fallback /data/processed/unified/manifests/plc_geometry/
python scripts/run_unified_preprocessing.py --raw-root /data/raw --output-root /data/processed/unified \
    --workers 8 --reuse-plc-geometry
```

Platform: the reference run was made on Windows 11. The code uses only portable Python, SimpleITK and nnU-Net calls and has no Windows-specific paths, but it has not yet been executed end to end on Linux.

## Stage 0: PLC-CECT geometry recovery

For each of the 1,444 released PLC volumes:

1. Read the CT and the released liver mask, map the 8-bit values to HU (`HU = v / 255 * 400 - 200`), set the in-plane spacing to 0.78125 mm.
2. Initial slice-spacing guess = 170 mm / (liver slices), 170 mm being the median liver craniocaudal extent of MCT-LTDiag and WAW-TACE. The guess only scales the model input; the final estimate does not depend on it. A released mask that spans less than 40 % of a volume is treated as a failed segmentation: a second pass uses 170 mm / (0.96 x volume slices) and the pass with the more consistent vertebra labels is kept.
3. Slice order. The liver area per slice tapers abruptly at the dome and slowly at the inferior tip; the sign of that asymmetry gives the stored order. The four phases of a patient are cross-checked by correlating their profiles (a phase counts as reversed relative to PVP only when the reversed correlation beats the direct one by 0.05) and the patient decision is the sum of the aligned margins. On MCT-LTDiag and WAW-TACE, whose headers are trusted, this rule is right in 80 of 80 cases.
4. The TotalSegmentator 3 mm model (task 297, fold 0, no mirroring) is run in-process through the nnU-Net predictor on the volume resampled to 3 mm in RAS voxel order (running the TotalSegmentator command line costs 60-150 s per volume on Windows because of process spawning; in-process it is about 1.5 s). The detected vertebrae are sorted along z and scored Kendall-style (+1 per pair whose levels ascend with z, lumbar below thoracic, -1 per inverted pair). Below a score of 4 the other slice order is segmented too and the more consistent one is kept. On the release 1,442 volumes are reversed; the profile rule was overridden by the vertebra order in 78 volumes (33 patients, mostly one phase stored in the other order).
5. Slice spacing per volume = sum of reference distances / sum of measured centroid distances (in slices) over consecutive vertebra pairs with adjacent levels that do not touch the volume edge. Adult reference distances between consecutive vertebral centroids (mm): T7-T8 23.5, T8-T9 24.5, T9-T10 25.5, T10-T11 27.0, T11-T12 29.0, T12-L1 31.5, L1-L2 33.5, L2-L3 35.0, L3-L4 35.5, L4-L5 35.0 (a generic 30 mm is used only when no level-specific pair exists).
6. Fusion per patient: the liver spans the same physical z range in every phase, so `spacing_phase = L / liver_slices_phase` with one shared `L`, the weighted median of the per-phase products. A phase whose own estimate (>= 2 level-specific pairs) disagrees with the shared value by more than 15 % keeps its own estimate (its scan or mask does not cover the same range; `spacing_z_source = vertebrae_this_phase_liver_extent_inconsistent`).
7. Confidence: `high` when at least two phases agree within 10 %, `medium` within 20 % or for a phase that keeps its own estimate, `low` otherwise.
8. Volumes whose released liver mask is empty or covers less than 60 % of the segmented liver get the TotalSegmentator liver written to `manifests/plc_geometry/liver_fallback/` and stage 1 uses it instead (25 volumes in the release).

Accuracy: individual stature adds about +/-8 % to population reference distances; within a patient the four phases agree to a median 8.5 % (max-min of the implied liver extents), so the per-volume noise is about +/-5 %. Expect +/-10 % on the slice spacing, about +/-3 % on an equivalent spherical diameter. Over the 1,438 volumes with a vertebra estimate the ratio to the nearest nominal reconstruction increment has median 1.01 (interquartile 0.97-1.06) with histogram peaks at 0.65-0.7, 0.8, 1.0, 1.45, 2.05 and 5.2-5.6 mm, so the reference table is unbiased at the population level and `--reference-scale` stays at 1.0. Estimates are kept continuous; the release mixes increments such as 0.3, 0.6, 0.65, 0.8, 1.0, 1.25, 1.5, 2, 2.5, 5 and 7.5 mm and the peaks overlap. Result: 0.29-8.05 mm (p5 0.64, median 1.19, p95 5.6); confidence high 724 / medium 660 / low 60 volumes.

Outputs: `plc_geometry.csv` (one row per volume: spacing, `k_axis_flip`, sources, confidence, liver extent, vertebra levels, fallback file as a path relative to `manifests/`), `plc_geometry/plc_vertebra_measurements.jsonl` (raw measurements, append-only, resumable), `plc_geometry/plc_geometry_summary.json`. The CSV contains no machine-specific paths, so `reference/plc_geometry.csv` can be dropped into another output root's `manifests/` folder to skip stage 0 (the fallback liver masks it refers to must then be regenerated or copied).

## Stage 1: conversion

| Dataset | Phase mapping, reference | Geometry source | Registration | Intensity |
|---|---|---|---|---|
| MCT-LTDiag | released NC / arterial / PVP / delay; PVP reference | released headers | released cross-phase correspondence (identity) | released HU |
| PLC-CECT | P, C1, C2, C3 -> NC, AP, PVP, DP; C2 / PVP reference | `plc_geometry.csv` | liver-mask rigid / affine + MI B-spline to PVP, Dice gate 0.80 | 8-bit window mapped back to [-200, 200] HU |
| WAW-TACE | phases 0-3 -> NC, AP, PVP, DP; PVP reference; tumour annotated in one phase (AP 90, PVP 48, DP 24, NC 2) and propagated to PVP | released headers | same as PLC-CECT | released HU |

Registration of each non-reference phase (PLC-CECT, WAW-TACE):

1. Initial transform: translation from the liver-mask centroids.
2. Rigid (`Euler3DTransform`) and affine (`AffineTransform`), both optimised with mean squares between the two liver masks smoothed with a 4 mm Gaussian on a grid of at least 4 mm inside the liver bounding box plus 30 mm, three resolution levels, regular-step gradient descent with physical-shift scaling. Because both phases have a liver mask this stage cannot collapse the way an intensity metric with a moving-image mask does. The affine candidate is accepted only when all three axis scales lie within 0.85-1.18: inside that range the z scale absorbs residual PLC slice-spacing error; outside it the two masks do not cover the same anatomy and the rigid candidate is used (`affine.rejected` in `case.json`).
3. B-spline: Mattes mutual information (32 bins, 20 % random samples inside the reference liver dilated by 10 mm, no moving mask) between the HU images clipped to [-200, 300], control-point spacing about 40 mm, L-BFGS-B, initialised from the best of the centroid, rigid and affine candidates.
4. Selection: the B-spline candidate is kept when its liver Dice is within 0.02 of its base; otherwise the highest liver Dice wins. Tumour masks never influence the transform.
5. Gate: `usable = final liver Dice >= 0.80`. Failed phases are still resampled and written but flagged in `case.json` (`registrations.<phase>.usable`) and in the manifest (`usable_nc`, `usable_ap`, `usable_dp`, `usable_phases`). For WAW-TACE the tumour mask is propagated from the annotated phase; `tumour_mask_reliable = 0` when that phase failed the gate.

Saved transforms: `resampling_transform_pvp_to_<phase>.h5` is the ITK transform used for resampling. It maps physical points of the PVP (fixed, output) grid to physical points of the native `<phase>` (moving) image; pass it to `sitk.Resample` to pull that phase onto the PVP grid. To move `<phase>` coordinates or landmarks into PVP space, invert it. `case.json` repeats this under `registrations.<phase>.transform_semantics`.

Output grid: the reference liver mask united with the tumour mask, padded by 20 mm, defines a box on the reference native grid; the output is a 1 mm grid whose origin is the physical position of that box's first voxel centre. Every channel is produced by **one** interpolation from its native grid through its transform onto this grid (linear for images, nearest neighbour for masks, `-1000` HU outside the acquired field). Images are rounded to int16. Because all outputs are LPS with an identity direction, the arrays can be used directly; the origin only places a case in its scanner's world coordinates, which is why different patients do not overlap in a viewer and why no cross-patient alignment is needed.

Sanity checks, recorded per case (`sanity.flags`, `sanity_pass`) and never blocking: reference field of view outside 250-500 mm; any phase slice spacing outside 0.4-7 mm; zero origin (non-PLC); liver volume outside 500-3,500 mL or empty; a phase whose liver z extent differs by more than 30 % from the reference phase; a phase below the Dice gate.

### Known intensity offset of PLC-CECT (recorded, not altered)

The PLC-CECT authors state that the release was windowed to [-200, 200] HU (level 0, width 400; paper and `window_adjust.py`), and the stored values follow that statement (`HU = v / 255 * 400 - 200`). Measured against the other two datasets, PLC tissues come out systematically lower:

| Tissue (median over 25-40 cases per dataset) | MCT-LTDiag | WAW-TACE | PLC-CECT | PLC offset |
|---|---:|---:|---:|---:|
| Non-contrast liver parenchyma, HU | 57 | 49 | 12 | about -41 |
| Portal-venous liver parenchyma, HU | 101 | 93 | 66 | about -31 |
| Portal-venous fat peak, HU | -101 | -95 | -133 | about -35 |

A non-contrast liver of 12 HU is not physiological (about 55 HU is), and fat shows the same shift, so the release was most likely windowed at level 40 ([-160, 240] HU) rather than level 0. Because the authors document level 0 and we have no DICOM to settle it, the stored voxels keep the authors' mapping. The offset is recorded instead: `case.json` carries `intensity_offset_hu_recommended` (40 for PLC-CECT, 0 for the other datasets) and `intensity_offset_basis`, and the manifest has the column `intensity_offset_hu_recommended`. A loader should add it before clipping:

```python
hu = image.astype(np.float32) + float(row["intensity_offset_hu_recommended"])
x = np.clip(hu, -200, 200) / 200.0          # same tissue -> same value in all three datasets
```

Without the offset a network trained on MCT-LTDiag and WAW-TACE still segmented unseen PLC-CECT livers with a mean Dice of 0.89 (a small 3D U-Net, 1,500 iterations), so the shift is not large enough to break mixed training. With the offset applied in the loader the median normalised liver intensity of PLC-CECT (NC / AP / PVP / DP: 0.26 / 0.30 / 0.44 / 0.46) coincides with MCT-LTDiag (0.27 / 0.36 / 0.52 / 0.43) and WAW-TACE (0.23 / 0.28 / 0.45 / 0.41), where before it was 0.06 / 0.10 / 0.24 / 0.26; the transfer Dice is unchanged (liver 0.89, tumour 0.35).

## Stage 2: labels and lesion sizes

`tumor_mask.nii.gz` is a semantic binary mask (0 background, 1 liver lesion). Patient-level targets in `case.json` and the manifest: lesion presence (absent / present), behaviour (benign / malignant / not applicable), origin (primary hepatic / extrahepatic metastatic / not applicable), coarse diagnosis (control, HH, HCC, ICC, cHCC-CCA, metastasis), fine diagnosis (CN, HH, HCC, ICC, cHCC-CCA, CRLM, BCLM); controls use -1 for behaviour and origin so those losses can be masked. A lesion is one 26-connected component on the 1 mm grid; its volume is converted to an equivalent spherical diameter (ESD) with bins 0-5, 5-10, 10-15 and > 15 mm (closed upper bounds); no minimum component size; ESD is not the RECIST longest diameter.

## Output layout

```text
<output-root>/
├── cases/
│   └── <DATASET>_<patient>/
│       ├── image_nc.nii.gz            int16 HU, 1 mm, LPS
│       ├── image_ap.nii.gz
│       ├── image_pvp.nii.gz
│       ├── image_dp.nii.gz
│       ├── liver_mask.nii.gz          uint8 {0, 1}
│       ├── tumor_mask.nii.gz          uint8 {0, 1}
│       ├── resampling_transform_pvp_to_<phase>.h5   PLC-CECT and WAW-TACE only
│       └── case.json
├── manifests/
│   ├── waw_four_phase_patients.csv
│   ├── plc_geometry.csv
│   └── plc_geometry/                  measurements, summary, liver_fallback/
├── dataset_manifest.csv
├── failures.json
├── summary.json
├── label_schema.json
├── lesion_size_statistics.csv
└── lesion_size_summary.csv
```

`case.json` records the pipeline version, reference phase, `usable_phases`, per-phase registration details (selected stage, Dice of every candidate, affine axis scales, optimiser stop conditions, errors, transform file), the native geometry of every phase, the PLC geometry provenance, the crop box, output geometry, intensity source and valid range, sanity results, liver volume and the archive members every input came from. `dataset_manifest.csv` has one row per case with the same information flattened (native slice spacing per phase, `spacing_source`, `spacing_confidence`, `intensity_valid_range_hu`, `intensity_offset_hu_recommended`, Dice and `usable_*` per phase, `tumour_mask_reliable`, `sanity_pass`, `sanity_flags`) plus the label and lesion-size columns.

## Patient-level split and size-binned evaluation

`scripts/make_splits.py` produces a reproducible patient-level train / validation / test split of a unified_v2 root and `scripts/evaluate_by_lesion_size.py` scores predicted tumour masks by lesion size. Both use the size bins of stage 2: equivalent spherical diameter of a 26-connected component on the 1 mm grid, **0-5, 5-10, 10-15 and > 15 mm**.

```bash
python scripts/make_splits.py --root /data/processed/unified --seed 0                        # small-held-out (default)
python scripts/make_splits.py --root /data/processed/unified --seed 0 --protocol stratified  # alternative
#   -> <root>/splits/unified_v2_<protocol>_seed0.{json,_cases.csv,_summary.csv}
python scripts/evaluate_by_lesion_size.py --root /data/processed/unified \
    --predictions /path/to/predictions --split /data/processed/unified/splits/unified_v2_small_held_out_seed0.json \
    --subset test --out /path/to/metrics
```

Two protocols, both deterministic (the same manifest, seed and options give the same subsets; the JSON records the case lists, per-label counts and options) and both patient-level (all phases and lesions of a patient stay together):

- **small-held-out (default, the project's protocol).** The question is whether a model that only ever learns from large lesions comes to find small ones. Every patient with at least one lesion in the 0-5, 5-10 or 10-15 mm bins (`--small-bins`) is kept out of training and goes to the test subset (`--small-val-fraction`, default 0, moves a share to validation). Patients whose lesions are all > 15 mm, and lesion-negative patients, are split 80 / 10 / 10 (`--ratios`) by iterative stratification over dataset, diagnosis and lesion presence, so validation exists for model selection on large lesions and the test subset also measures large-lesion performance. The training set contains no lesion <= 15 mm (smallest training lesion in the reference split: 15.04 mm ESD).
- **stratified.** Every stratum (dataset, coarse diagnosis, lesion present / none, `has_<bin>` for each size bin and the joint `dataset:has_<bin>`) is represented in every subset in proportion to `--ratios` (default 70 / 10 / 20), using iterative stratification (Sechidis et al. 2011) with the rarest label placed first. Use it when the model should also train on small lesions.

Clean evaluation subsets (default): cases with `tumour_mask_reliable = 0` or with a phase below the registration gate never enter validation or test. Large-only ones are forced into train, where phase dropout handles them; under small-held-out a flagged small-lesion patient can neither train nor be evaluated and is excluded (WAW-TACE 33 and 34 in the reference split). `--no-clean-eval` disables this.

Reference splits (`reference/splits/`, seed 0):

| Protocol | Train | Val | Test | Small-lesion patients in test | Test lesions per bin (0-5 / 5-10 / 10-15 / > 15 mm) |
|---|---:|---:|---:|---:|---|
| small-held-out | 534 | 67 | 439 | 372 (all) | MCT 508 / 458 / 323 / 645; PLC 1 / 9 / 23 / 49; WAW 0 / 8 / 19 / 48 |
| stratified | 727 | 105 | 210 | 78 | MCT 91 / 57 / 61 / 182; PLC 0 / 2 / 3 / 58; WAW 0 / 2 / 4 / 47 |

Under small-held-out the training set keeps 144 of the 517 MCT-LTDiag patients (the rest carry small lesions), 279 PLC-CECT and 111 WAW-TACE patients. Small lesions come almost entirely from MCT-LTDiag, whose native slices are 5 mm: a sub-5 mm component there is a one- or two-slice object, and the 0-5 mm bin should be read with that in mind; PLC-CECT contributes 33 lesions <= 15 mm and WAW-TACE 27.

Evaluation rules (`evaluate_by_lesion_size.py`): predictions are binary NIfTI files `<case_id>.nii.gz` on the case grid. A ground-truth lesion counts as detected when at least `--hit-fraction` (default 0.10) of its voxels are covered by the prediction; a predicted component that touches no lesion is a false positive. Reported per size bin, overall and per dataset: lesions, detected, sensitivity with a 95 % Wilson interval, mean lesion-wise Dice (0 for a miss), false-positive components in that size range; per case: tumour Dice and false positives per case. Outputs `size_binned_metrics.json`, `size_binned_metrics.csv` and `per_lesion.csv`.

Worked example (`reference/splits/example_size_binned_metrics_stratified.csv`, stratified protocol): a deliberately small 3D U-Net (16-128 channels, 96 mm patches, 1,500 iterations on 150 of the 727 training cases, no post-processing) predicted the 210 test cases; the evaluator then gave, over all datasets, sensitivity 0.08 [0.04, 0.15] for 0-5 mm (91 lesions), 0.07 [0.03, 0.16] for 5-10 mm (61), 0.22 [0.14, 0.33] for 10-15 mm (68) and 0.67 [0.61, 0.72] for > 15 mm (287), with 31 false-positive components per case. These numbers only illustrate the output format and the size dependence; they are not a result of this repository.

## Using the output as one network input

- Load the four channels, add the manifest's `intensity_offset_hu_recommended` (40 HU for PLC-CECT, see above), clip to at most [-200, 200] HU (the PLC valid range; MCT / WAW are unclipped in storage) and normalise in the loader. Everything is already 1 mm isotropic, LPS, liver-centred; array sizes differ per case, so pad or crop in the loader.
- Honour `usable_nc/ap/dp`: zero or drop a flagged channel (phase dropout) rather than training on a misaligned one. PVP is always usable.
- Honour `tumour_mask_reliable` (WAW-TACE) and `sanity_pass` when selecting training cases.
- For size-stratified evaluation use the native slice spacing (`native_spacing_z_*_mm`) and `spacing_confidence`: a 5 mm native slice cannot resolve a 3 mm lesion whatever the output grid says, and PLC sizes carry the estimation uncertainty.
- Make patient-level splits after preprocessing; all phases and lesions of one patient stay together.

## Reference outputs and checks

Numbers obtained on 2026-09-24 from the current releases (also in `reference/`):

| Dataset | Cases | Lesion-positive | Lesions | ESD 0-5 / 5-10 / 10-15 / > 15 mm | All four phases usable | Sanity flags |
|---|---:|---:|---:|---|---:|---:|
| MCT-LTDiag | 517 | 517 | 2,159 | 508 / 458 / 323 / 870 | 517 | 4 |
| PLC-CECT | 361 | 278 | 343 | 1 / 9 / 23 / 310 | 344 | 51 |
| WAW-TACE | 164 | 164 | 262 | 3 / 9 / 19 / 231 | 162 | 10 |
| Combined | 1,042 | 959 | 2,764 | 512 / 476 / 365 / 1,411 | 1,023 | 65 |

Liver Dice after the selected transform (median NC / AP / DP, phases below the 0.80 gate): PLC-CECT 0.94 / 0.96 / 0.97 (16 / 6 / 3); WAW-TACE 0.96 / 0.97 / 0.97 (2 / 2 / 2, the two whole-body scans 33 and 34 whose released organ masks are inconsistent with their CT). On the nominal 1 mm grid of the release PLC-CECT appeared to contain dozens of sub-15 mm lesions; on the recovered geometry 33 of 343 components are <= 15 mm.

`reference/` holds `plc_geometry.csv` (per-volume slice order and spacing), `liver_fallback/` (the 29 TotalSegmentator liver masks used where the released PLC liver masks were empty or truncated, 3 MB), `plc_geometry_summary.json`, `dataset_manifest.csv` (one row per case with every quality field), `lesion_size_summary.csv` and `summary.json`. A rerun should reproduce the counts above exactly and the PLC spacings to within numerical noise (registration uses fixed random seeds; TotalSegmentator inference is deterministic on one GPU model but may differ in the last digit across GPU generations).

Checks after a run:

```bash
python -m unittest discover -s tests -v                       # conventions, runner, labels
python scripts/verify_unified_v2_output.py --root /data/processed/unified
```

The verifier reads every case (six files present, identical 1 mm grids, int16 images, binary uint8 masks, current pipeline version) and prints per-dataset counts, usable-phase sets and sanity-flag kinds; `summary.json` and `failures.json` give the completion and failure inventory.

## Limitations

- PLC-CECT slice spacing is an anatomical estimate (+/-10 %) and the slice-order correction is inferred, not documented by the authors (GitHub issues #3 and #7 of `ljwa2323/PLC_CECT` had no answer at the time of writing); PLC intensities above 200 HU are saturated by the release, and the stored PLC values are about 40 HU below the other datasets under the authors' stated window (recorded as `intensity_offset_hu_recommended`, see Stage 1).
- MCT-LTDiag was acquired at 5 mm slices; its 1 mm grid is interpolated. WAW-TACE mixes 0.44-7.5 mm native slices.
- Registration quality is measured on the liver boundary only; a passed gate does not guarantee millimetre alignment of small lesions across phases.
- The three datasets have different populations and annotation protocols (WAW-TACE: HCC only, TACE setting); a shared mask encoding does not make their disease distributions identical.

## Repository layout

```text
scripts/run_unified_preprocessing.py    stages 0-2 in order, dataset selection, per-dataset roots, dry run, resume, replace
scripts/estimate_plc_geometry.py        stage 0 (download-weights / measure / solve / all)
scripts/preprocess_multiphase_ct.py     stage 1
scripts/prepare_hierarchical_labels.py  stage 2
scripts/verify_unified_v2_output.py     read-only completion check
scripts/make_splits.py                  reproducible stratified patient-level split
scripts/evaluate_by_lesion_size.py      size-binned lesion detection / segmentation metrics
tests/                                  unit tests (unittest)
reference/                              outputs of the reference run, incl. splits/
requirements.txt                        minimum versions
requirements-lock.txt                   exact versions of the reference environment
```
