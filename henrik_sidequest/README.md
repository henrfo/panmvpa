# henrik_sidequest — PAN-MVPA

Exploring the benefit of precision fMRI for multivariate analysis on
[OpenNeuro ds006598 (PAN)](https://openneuro.org/datasets/ds006598/versions/1.0.0).

## Verified dataset facts (checked against S3, 2026-07-20)

- 10 subjects `sub-PAN01`..`sub-PAN10`, up to 9 sessions each.
- Preprocessed fMRIPrep BOLD sits in the **subject/session func folders** (not a separate
  derivatives tree): `sub-PAN{XX}/ses-{N}/func/*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz`
- **TR = 1.355 s**, slice-timing NOT corrected, skull-stripped, MNI152NLin6Asym 2mm.
- One `epiproj` run per session (no `run-` label) → **session = run** for cross-validation.
- AFNI timing: `derivatives/afni_timing/PAN{XX}/sub-PAN{XX}_ses-{N}_task-epiproj_{cond}.1D`
  - Conditions: `pastself, presentself, futureself, pastnonself, presentnonself, futurenonself`
  - Format: one row of onset times in **seconds**. **No durations** — set in `config.py`.
- **No fMRIPrep confounds .tsv are deposited.** First-level GLM uses cosine drift only.

## Data access

Full dataset is ~636 GB. Clone metadata, then `datalad get` only what a step needs.

```bash
datalad clone https://github.com/OpenNeuroDatasets/ds006598.git data/ds006598
python fetch_data.py --subject PAN01 --task epiproj   # datalad get just this slice
```

Set `PANMVPA_DATA` to point at the clone (defaults to `henrik_sidequest/data/ds006598`).

## Steps

1. **First-level GLM** (`glm.py`, `run_glm.py`) — per-session epiproj GLM, one beta map per
   condition per session. ← current
2. Parcellations — Schaefer-400/Yeo-17 template atlases (later).
3. MVPA — decode retrospection (past vs present) vs prospection (future vs present) within
   DN-A parcels; compare template vs group vs individual (later).
