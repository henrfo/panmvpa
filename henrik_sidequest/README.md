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

- **Plot 1 — map stability.** At each data level, split the rest in half, build a
  parcellation from each half, Dice-overlap the two DN-A masks.
- **Plot 2 — subject identification.** At each data level, build all 10 subjects' maps,
  then take a held-out scan and score it against every map by within-network homogeneity
  (mean voxel-voxel correlation within a network, averaged over the 17 networks). The
  best-fitting map is the prediction. Chance = 1/10.

Each level is repeated over `--n-seeds` random subsets of rest runs, so every subject
gets error bars and a noisy subject can be told apart from a flat effect.

The x-axis is capped at **100 min** so all 10 subjects contribute at every level (PAN03
and PAN05 have only ~105 min of rest, PAN07 ~115).

### Design notes worth knowing

- **Held-out scans are task runs only.** Task runs are never used to build maps at any
  level, so the test set is *identical* along the whole x-axis. Leftover rest runs would
  change with level and seed, confounding "better maps" with "different test data".
- **Homogeneity rises as networks shrink**, so a map that carves smaller networks could
  win for the wrong reason. A size-matched control (each network subsampled to a common
  voxel count) is reported alongside; the two should agree. `--no-size-matched` skips it.
- **Seed spread is not comparable across the x-axis.** At 20 min a seed draws 4 runs from
  a pool of 21-33; at 100 min it draws 20 from 21-24, so PAN03/PAN05 have only *one*
  spare run and their seeds are near-identical. Error bars therefore narrow toward
  100 min partly by construction. `spare_runs` is reported per level and the CLI warns
  when it is <= 1.

Verified: on a held-out PAN01 task scan, homogeneity under PAN01's own 40 min map
(0.0498) beats the group Yeo-17 atlas (0.0354), which beats a size-preserving shuffle
(0.0150). A structureless-noise control lands on chance (0.245 over 400 scans, chance
0.250), so the scorer carries no size bias.

Earlier Plot 2 designs (4-class task decoding, DN-A contrast-to-noise) live on in
`mvpa.py` and `cnr.py`; `run_all.py` no longer calls them.

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
- `panmvpa/identify.py` — within-network homogeneity, subject identification (Plot 2)
- `panmvpa/mapstore.py` — persist maps so raw scans can be deleted after pass 1
- `panmvpa/cnr.py` — DN-A contrast-to-noise (superseded, retained)
- `panmvpa/mvpa.py` — 4-class SVM decoding (retained, not run by `run_all.py`)
- `panmvpa/figure.py` — two-panel group figure (mean ± SEM) + CSV
- `panmvpa/atlases.py` — download Schaefer-400/Yeo-17 into `atlases/`
- `scripts/run_all.py` — end-to-end: both curves, PNG + CSV
- `fetch_data.py` / `verify.py` — data fetch + smoke test
- `atlases/` — template parcellations, committed (~3 MB)

## Running it

```bash
panmvpa-run                          # all 10 subjects -> group figure
panmvpa-run --subjects PAN01         # single subject
python scripts/run_all.py            # same thing, without installing
```

Writes `results/group_figure.png` (mean across subjects ± 1 SEM, with faint per-subject
traces), `group_curves.csv` (per-subject rows + GROUP_MEAN / GROUP_SEM) and
`group_results.json`. `results/` is tracked in git so hub output can be pulled back.
Subjects lacking enough rest or any fetched epiproj run are skipped with a warning
rather than aborting the run.

### Where the data lives

`DATA_DIR` (or `PANMVPA_DATA`) points at the ds006598 root, so the same code runs
locally and on a hub with no edits. It falls back to the in-repo `data/ds006598` clone.

## Running on JupyterHub

The whole cohort is 835 runs / **0.64 TB**, which need never be on disk at once. Fetch
one kind of data at a time and let `--cleanup` delete each subject's raw scans once
their numbers are computed — disk stays around one subject's worth (~20-60 GB).

```bash
git clone <repo-url> && cd panmvpa && git checkout henrik_sidequest/precision-mvpa
pip install -e .                       # installs henrik_sidequest/panmvpa
export DATA_DIR=$HOME/data/ds006598
F=henrik_sidequest/scripts/fetch_hub.py

# Pass 1 — rest only: Dice curve + build/save every map, then drop the rest data.
for S in PAN01 PAN02 PAN03 PAN04 PAN05 PAN06 PAN07 PAN08 PAN09 PAN10; do
  python $F --dest $DATA_DIR --subjects $S --kind rest
  panmvpa-run --stage maps --subjects $S --cleanup
done

# Pass 2 — held-out task scans, scored against ALL subjects' stored maps.
for S in PAN01 PAN02 PAN03 PAN04 PAN05 PAN06 PAN07 PAN08 PAN09 PAN10; do
  python $F --dest $DATA_DIR --subjects $S --kind task --max-tasks 12
  panmvpa-run --stage identify --subjects $S --cleanup
done

panmvpa-run --stage figure
git add results/ && git commit -m "hub results" && git push
```

**Why two passes.** Scoring a scan needs *every* subject's map, so a single
download-process-delete pass would destroy PAN01's scans before PAN10's map exists. Maps
are the durable artefact (258 KB each; the full 10x5x10 cohort is ~130 MB) and raw BOLD
is transient. Pass 2 can therefore score any scan against everyone long after the scans
are gone. Both stages are resumable: existing maps are not rebuilt and
`identification.json` accumulates.

`--cleanup` **permanently deletes** the raw BOLD it has finished with. Everything is
re-downloadable from S3, but do not point it at a dataset you curate by hand.

`fetch_hub.py` pulls from OpenNeuro's public S3 over HTTPS — stdlib only, no datalad or
git-annex. `--kind rest|task|all` splits the two passes, `--max-tasks N` caps held-out
runs per subject spread evenly across task families, and files already present at the
right size are skipped so an interrupted fetch resumes.

**Memory.** The hub's 15 GB is the binding constraint. Subjects are processed one at a
time and every cache is dropped (and `gc.collect()`ed) between them, so peak RSS tracks a
single subject, not the cohort. **Measured peak: 7.7 GB for one subject at the 100 min
level** (PAN01, 132k-voxel domain) — it fits in 15 GB but not by a huge margin. Peak RSS
is printed after each subject; watch the first one.

If it is tight, shrink the rest-run cache — this trades re-reads for headroom and costs
roughly 120 MB per cached run:

```bash
export PANMVPA_REST_CACHE=6     # default is 10
```

The dominant costs are the concatenated timeseries at the top data level (~2.3 GB) and
nilearn's GLM fit; both scale with the number of domain voxels and TRs, not with cohort
size.

**Long runs.** Enable the Jupyter keepalive plugin (`cmd-shift-C`, search "keep", set
24 h) — a full 10-subject run takes well over an hour.

Note the repo root installs the package from `henrik_sidequest/panmvpa`; the driver is
importable as `panmvpa.cli`, which is why `panmvpa-run` works from any directory.

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
