# henrik_sidequest — PAN-MVPA plumbing

Data plumbing for the PAN precision-fMRI MVPA sidequest on
[OpenNeuro ds006598 (PAN)](https://openneuro.org/datasets/ds006598/versions/1.0.0).
This is infrastructure only — no GLM, no MVPA yet. The point is to `import panmvpa` in a
notebook and start exploring.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install numpy scipy pandas nibabel nilearn scikit-learn matplotlib
```

## Use

```python
import panmvpa

panmvpa.epiproj_sessions("PAN01")      # -> [1, 2, 4, 5, 6, 7]  (only real epiproj sessions)
ev = panmvpa.build_events("PAN01", 1)  # onset / duration(=10s) / trial_type
img = panmvpa.load_bold("PAN01", 1)    # nibabel image, MNI152NLin6Asym 2mm
```

## Package layout

- `panmvpa/config.py` — paths, `SUBJECTS`, `TASK`, `TR`, `CONDITIONS`, durations, path helpers
- `panmvpa/events.py` — `parse_1d_file`, `build_events`, `epiproj_sessions`
- `panmvpa/bold.py` — `find_bold`, `load_bold`, `is_fetched`
- `panmvpa/atlases.py` — download Schaefer-400/Yeo-17 into `atlases/`
- `fetch_data.py` — `datalad get` only the epiproj slice for a subject
- `verify.py` — smoke test (events table + BOLD shape + atlases)
- `atlases/` — template parcellations, committed (~3 MB)

## Getting the data

Full dataset is ~636 GB; one epiproj BOLD run is ~1.3 GB. Clone metadata, fetch per step.

```bash
datalad clone https://github.com/OpenNeuroDatasets/ds006598.git data/ds006598
python fetch_data.py --subject PAN01     # datalad get epiproj BOLD + timing only
python verify.py --subject PAN01
```

`datalad get` needs the `git-annex-remote-openneuro` helper on PATH. Set `PANMVPA_DATA`
if your clone lives elsewhere (defaults to `henrik_sidequest/data/ds006598`).

## Verified dataset facts (S3 + paper STAR Methods, 2026-07-20)

- 10 subjects `PAN01`..`PAN10`; **~6 epiproj runs per subject**, spread across sessions
  (e.g. PAN01 → sessions 1,2,4,5,6,7). Never assume a session has epiproj — discover it.
- Preproc BOLD lives in the subject/session func folders (not a separate derivatives tree):
  `sub-PAN{XX}/ses-{N}/func/*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz`
- **TR = 1.355 s**, slice-timing NOT corrected, skull-stripped.
- epiproj block = 20 s (5 s fix + **10 s trial** + 5 s fix). We model the 10 s trial.
- Conditions: `pastself, presentself, futureself, pastnonself, presentnonself, futurenonself`.
- AFNI `.1D`: one row of onset times (s); `*` = empty run. **No durations in the file.**
- **No fMRIPrep confounds .tsv are deposited.**

## Notes for the analysis phase (not done here)

- Yeo-2011 atlas ships in FreeSurferConformed **1 mm** space — resample to the BOLD grid
  (MNI152NLin6Asym 2 mm) before masking. Schaefer ships in FSL MNI152 2 mm.
- Contrasts of interest: retrospection (past vs present self), prospection (future vs present).
