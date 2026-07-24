# panmvpa — how much rest data does a personal brain map need?

**Stable** = the map gives the same answer every time.
**Useful** = the map identifies whose brain a scan came from.

Data: [OpenNeuro ds006598 (PAN)](https://openneuro.org/datasets/ds006598/versions/1.0.0),
10 subjects, preprocessed fMRIPrep BOLD in MNI152NLin6Asym 2mm. Volumetric throughout —
no surface, no CIFTI, no FreeSurfer.

## Building a personal map

The Yeo-17 group atlas is a fixed spatial anchor: it is regridded once (nearest-neighbour)
to the BOLD grid and collapsed to 17 network regions. Those regions never move — not
across data levels, not across subjects. Then, from a subject's rest data:

1. Average the timeseries within each of the 17 group regions → **17 reference signals**.
2. Correlate every cortical voxel against all 17 references.
3. Assign each voxel to its best match — winner-take-all. That's the personal map.

What changes with more data is the *reference signals* (computed from more rest) and
therefore the voxel assignments. What never changes is where the 17 group regions are.

> The Yeo-17 taxonomy is realised via Schaefer-400, whose parcels carry the Yeo-Krienen
> 17-network labels and which ships in FSL-MNI152 2mm — the same space family as the BOLD,
> so it needs only a regrid rather than a cross-space resample.

## Increasing the data — deterministic, no random seeds

Each subject's rest runs are taken in order and split into four equal quarters. Eight maps
are built:

| maps | data each | purpose |
|---|---|---|
| Q1, Q2, Q3, Q4 | a quarter | stability at 1/4 (6 pairs) |
| Q1+Q2, Q3+Q4 | a half | stability at 2/4 (1 pair) |
| Q1+Q2+Q3 | three quarters | identification at 3/4 |
| Q1+Q2+Q3+Q4 | everything | identification at 4/4 |

## Plot 1 — stability

Compare maps built from the **same amount of data**: mean pairwise Dice across the 17
networks. Higher = lower estimation variance.

Only 1/4 and 2/4 have two or more equal-sized maps, so **only those two levels have a
point**. Three-quarter and full are single maps — there is nothing to compare them with.

## Plot 2 — signal

Use the **cumulative** map at each level (Q1, Q1+Q2, Q1+Q2+Q3, all). Score every held-out
task scan against every subject's map by within-network homogeneity — the average
correlation between voxels a map groups together, averaged over the 17 networks. The
best-fitting map is the prediction; accuracy is the fraction correct. Chance = 1/10.

Computed with the identity `sum of pairwise correlations = ||sum of rows||^2 / T - n`, so
each network costs O(n·T) and no voxel-by-voxel correlation matrix is ever built.

## Fixed vs varying

| fixed at every level | varies with level |
|---|---|
| the 17 group regions (where they are) | the 17 reference signals |
| the analysis domain (132,032 voxels) | the voxel assignments |
| the held-out task scans | the resulting personal map |
| the scoring method | |

Held-out scans are **task runs only** — never used to build a map at any level — so the
test set is identical across the whole x-axis. The scan list is a function of subject
alone; it takes no level argument.

## Leaner variant — parcel covariance (`scripts/run_fc.py`)

Same two questions, the standard method for them: reduce each rest run to a **Schaefer-400
parcel covariance** and ask (1) how fast it converges and (2) how much rest a linear SVM
needs to tell the ten subjects apart. This is the ordinary FC-convergence + fingerprinting
approach — not a reimplementation of anyone's parcellation procedure.

**Reduce, then delete.** The download is 180 GB and the hub has 15, so the only real
machinery is a loop that pulls one run, shrinks it, deletes it. Each run becomes three
arrays (~300 KB vs 730 MB), computed with nilearn maskers:

```
parcels  (T, 400)  Schaefer-400 parcel means, RAW
gs       (T,)      whole-brain mean signal (brain mask, NLin6Asym — our exact grid)
dvars    (T,)      frame-to-frame RMS change — the motion-spike proxy
```

Raw on purpose: detrend / band-pass / global-signal regression are all linear and commute
with parcel-averaging, so cleaning is a cheap analysis-time knob —
`nilearn.signal.clean(parcels, confounds=gs, detrend, low_pass=0.08, high_pass=0.009,
standardize="zscore_sample")`. Z-scoring is the one non-linear step, so it happens *after*
averaging, never in the reduction. The dataset ships only preprocessed BOLD — no confounds,
no motion parameters, no masks — so the global signal is the nuisance lever we have.

**The main result — nearest-neighbour identification vs data (minutes of rest, linear):**

A raw within-person convergence curve is meaningless alone, because the scale isn't 0–1.
Two halves of *one* person's rest already agree ~0.9; two *different* people ~0.6 — most of a
connectivity table is just "this is a human cortex," and the between-person floor itself
**rises with data**. So everything grows on one ladder from the identical A-side estimate,
changing only the reference. For subject A at *X* minutes, correlate A's growing-half FC
against every subject's full reference half:

- **r_self(X)** — vs A's own reference (the convergence / reliability curve).
- **nearest(X)** — the **max** over other subjects: the nearest impostor, the identification
  competitor. **floor(X)** — the **mean** over others: the group floor (same cross-
  correlations, one loop).
- **signal(X) = r_self − nearest** — the individual signal, formed per subject then averaged
  and reported **directly** (not as headroom = signal/(1−nearest); that denominator moves, so
  headroom can rise while the signal itself falls). Unlike SVM accuracy it has **no ceiling**,
  and needing no held-out examples it runs the full ladder to 80 min.
- **hit rate** — was r_self the top match of all subjects? Reported, but it ceilings like the
  SVM, so it isn't the headline.

The n<10 tail (at 80 min, a single subject with the most rest) is **de-emphasised** in the
figure — bold only over the full-cohort rungs, the sparse tail greyed — so the highest point
on the chart isn't one person.

- **SVM (second method, `--svm`)** — one example = *X* minutes labelled by subject; grow *X*,
  retrain, record the **margin**. Held out by **whole session**, never random minutes; examples
  session-disjoint. An example eats *X* minutes, so it stops past ~40 min when each subject has
  one example to hold out. A **connectivity-free control** (per-parcel temporal mean/SD) is
  scored alongside it. It is the only slow part (20+ min), so it is **off by default** — pass
  `--svm` to run it; the curves, CSVs and network breakdown finish in seconds without it.

**`analyze` prints numbers and writes files — no prose conclusions** (a conclusion in prose
survives changes to the metric it came from; a CSV doesn't). It prints the subject-mean table
(r_self / nearest / floor / signal / hit%) and the SVM table, and writes:

- `curves.csv` — long format, one row per (subject, minutes): `subject, minutes, n, r_self,
  nearest, floor, signal`. The **per-subject** curves behind the means; 90% minutes, slopes,
  crossover are all computed from this in a notebook.
- `networks.csv` — long format per network block: `network_a, network_b, minutes, r_self,
  nearest, signal` (subject-mean per Yeo-17 pair, every rung).
- `curves.png` — two panels, shared linear x, thin line per subject behind each bold mean, tail
  greyed: **top** r_self and the group floor with the gap shaded (nearest impostor a thin
  reference); **bottom** the individual signal.
- `networks.png` — the mean reference FC with the 400 parcels **sorted by Yeo-17 network**
  (blocks line up with named systems) beside the 17×17 signal-per-block matrix at the last
  full-cohort rung.

```bash
python scripts/run_fc.py inspect --subjects PAN01           # reduce ONE run, look, delete nothing
python scripts/run_fc.py reduce  --subjects PAN01 --cleanup # reduce all rest, then drop the BOLD
python scripts/run_fc.py analyze                            # tables + CSVs + figures (seconds)
python scripts/run_fc.py analyze --svm                      # also the leave-one-session-out SVM (20+ min)
```

`inspect` first: deletion is the only irreversible step, and `--cleanup` skips any run whose
sanity check fails. The brain mask auto-downloads from templateflow on first run. nilearn's
per-run deprecation notices (confound standardization, masker resampling — harmless) are
silenced in-code, scoped to each nilearn call and class-agnostic, so the tables stay
readable without a stderr redirect.

**On the hub — prove it on one subject before looping over ten.** `--cleanup` deletes BOLD;
do not point it at all ten until PAN01 has gone through inspect → reduce and you have
confirmed the `.npz` files landed and the curve looks sane.

```bash
export DATA_DIR=$HOME/data/ds006598
F=henrik_sidequest/scripts/fetch_hub.py
R=henrik_sidequest/scripts/run_fc.py

# 1. One subject, end to end. STOP and look before trusting --cleanup on the cohort.
python $F --dest $DATA_DIR --subjects PAN01 --kind rest
python $R inspect --subjects PAN01                    # eyeball the diagnostic PNG
python $R reduce  --subjects PAN01 --cleanup
ls henrik_sidequest/derivatives/reduced/v1/           # confirm the .npz landed
python $R analyze                                     # one subject -> within only; curve sane?
```

```bash
# 2. Only once that looks right: the cohort. `set -e` stops at the first failure so an empty
#    DATA_DIR can't charge through all ten repeating the same error. Re-running is safe --
#    an already-reduced run is skipped, not rebuilt.
set -e
for S in PAN01 PAN02 PAN03 PAN04 PAN05 PAN06 PAN07 PAN08 PAN09 PAN10; do
  python $F --dest $DATA_DIR --subjects $S --kind rest
  python $R reduce --subjects $S --cleanup
done
python $R analyze
```

## Layout

```
panmvpa/config.py        paths, subjects, the quarter/level design
panmvpa/rest.py          find scans, split into quarters, load timeseries
panmvpa/parcellation.py  WTA map building, saving, map-to-map Dice
panmvpa/identify.py      homogeneity scoring and subject identification
panmvpa/figure.py        the two-panel plot + CSV
panmvpa/cli.py           stage runner (WTA maps)
scripts/fetch_hub.py     S3 streaming download, one subject at a time
scripts/run_all.py       entry point (WTA maps)
scripts/run_fc.py        the leaner parcel-covariance variant (reduce / inspect / analyze)
```

## Running it

```bash
pip install -e .                 # from the repo root
export DATA_DIR=$HOME/data/ds006598
```

Two passes, because scoring needs *every* subject's map — a single
download-process-delete pass would destroy early subjects' scans before later maps exist.

```bash
F=henrik_sidequest/scripts/fetch_hub.py

# Pass 1 — rest only: build maps, measure stability, drop the rest data.
for S in PAN01 PAN02 PAN03 PAN04 PAN05 PAN06 PAN07 PAN08 PAN09 PAN10; do
  python $F --dest $DATA_DIR --subjects $S --kind rest
  panmvpa-run --stage maps --subjects $S --cleanup
done

# Pass 2 — held-out task scans, scored against all 10 subjects' maps.
for S in PAN01 PAN02 PAN03 PAN04 PAN05 PAN06 PAN07 PAN08 PAN09 PAN10; do
  python $F --dest $DATA_DIR --subjects $S --kind task
  panmvpa-run --stage identify --subjects $S --cleanup
done

panmvpa-run --stage figure
```

Maps (~260 KB each) and the JSON/CSV results persist; raw BOLD streams through. Disk holds
about one subject at a time. Both stages resume — existing maps are not rebuilt and the
results files accumulate.

`--cleanup` **permanently deletes** the raw BOLD it has finished with. Everything is
re-downloadable from OpenNeuro. It handles both a git-annex clone (`annex drop`) and a
plain download (unlink).

### JupyterHub notes

- 15 GB RAM: subjects are processed one at a time and caches are dropped between them.
- `export TMPDIR=$HOME/tmp` — the overlay filesystem is small.
- Enable the keepalive plugin (`cmd-shift-C`, search "keep", 24 h) for long runs.

## Grid cache (run this if `compare` says "No BOLD on disk")

`compare`, `identify` and `figure` only read `.npy` maps, but the voxel grid used to be
derived from a BOLD header — so after `--cleanup` deleted the scans they crashed. The
grid is now cached:

```bash
python henrik_sidequest/scripts/build_grid_cache.py --verify-against-maps
```

This rebuilds the analysis domain and group map **from the atlas alone**, writing
`derivatives/domain.npy` and `derivatives/group_map.npy`. `--verify-against-maps` checks
the rebuilt domain size against the maps already on disk before you rely on it.

The `maps` stage writes this cache automatically on first run, so a fresh pipeline never
hits the problem. The geometry is hardcoded (MNI152NLin6Asym 2mm, 91×109×91) and was
verified byte-for-byte identical to the BOLD-derived domain.
