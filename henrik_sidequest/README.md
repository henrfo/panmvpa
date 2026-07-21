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

## The study

How much resting-state data do you need before an *individualised* brain map actually
helps classify what task someone is doing? Two curves, one figure, shared x-axis =
minutes of rest used to build the parcellation.

- **Plot 1 — DN-A reliability.** At each data level, split the rest in half, build a
  parcellation from each half, Dice-overlap the two DN-A masks.
- **Plot 2 — DN-A contrast-to-noise.** At each data level, build one parcellation and
  measure how far the epiproj signal inside DN-A sits above the rest of cortex:
  `CNR = mean(Z inside DN-A) − mean(Z in all other network voxels)`.

The x-axis is capped at **100 min** so all 10 subjects contribute at every level (PAN03
and PAN05 have only ~105 min of rest, PAN07 ~115).

### Why CNR rather than a classifier

Plot 2 was originally 4-class task decoding (language / ToM / epiproj / control). That
requires sessions containing all four tasks, and only PAN01/PAN02 have 4 such sessions —
six subjects have just 2 (8 samples for a 4-class SVM). CNR needs only the epiproj runs
that every subject has: no classifier, no cross-validation, no session-overlap
requirement. `mvpa.py` is retained for revisiting classification on PAN01/PAN02 as a
supplementary analysis; `run_all.py` does not call it.

### Method

The **group Yeo-17 atlas is a fixed anchor**. Schaefer-400 (whose parcels carry the
Yeo-Krienen 17-network labels, already in FSL-MNI152 2mm) is resampled once
(nearest-neighbour) to the BOLD grid and collapsed to 17 network *seed regions*. Those
region definitions never change with data amount and are never re-derived from the
individual. At each level the 17 reference signals are the mean timeseries within those
fixed regions computed from that level's rest data, and every cortical voxel is reassigned
by winner-take-all to its most-correlated reference. Only voxel allegiance is
individualised. **DN-A := DefaultC.**

## Package layout

- `panmvpa/config.py` — paths, subjects, TR, durations, minute levels, decoding families
- `panmvpa/events.py` — `parse_1d_file`, `build_events`, `epiproj_sessions`
- `panmvpa/bold.py` — `find_bold`, `load_bold`, `is_fetched`
- `panmvpa/rest.py` — enumerate rest runs, concatenate to a target number of minutes
- `panmvpa/parcellation.py` — WTA to the fixed group Yeo-17, `dn_a_mask`
- `panmvpa/reliability.py` — split-half Dice at each data level (Plot 1)
- `panmvpa/glm.py` — first-level GLM for any task -> task-vs-baseline beta / z-map
- `panmvpa/cnr.py` — DN-A contrast-to-noise during epiproj (Plot 2)
- `panmvpa/mvpa.py` — 4-class SVM decoding (retained, not run by `run_all.py`)
- `panmvpa/figure.py` — two-panel group figure (mean ± SEM) + CSV
- `panmvpa/atlases.py` — download Schaefer-400/Yeo-17 into `atlases/`
- `scripts/run_all.py` — end-to-end: both curves, PNG + CSV
- `fetch_data.py` / `verify.py` — data fetch + smoke test
- `atlases/` — template parcellations, committed (~3 MB)

## Running it

```bash
python scripts/run_all.py                      # all 10 subjects -> group figure
python scripts/run_all.py --subjects PAN01     # single subject
```

Writes `derivatives/figures/group_figure.png` (mean across subjects ± 1 SEM, with faint
per-subject traces) and `group_curves.csv` (per-subject rows + GROUP_MEAN / GROUP_SEM).
Subjects lacking enough rest or any fetched epiproj run are skipped with a warning
rather than aborting the run.

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
