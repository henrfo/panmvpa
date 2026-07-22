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

## Layout

```
panmvpa/config.py        paths, subjects, the quarter/level design
panmvpa/rest.py          find scans, split into quarters, load timeseries
panmvpa/parcellation.py  WTA map building, saving, map-to-map Dice
panmvpa/identify.py      homogeneity scoring and subject identification
panmvpa/figure.py        the two-panel plot + CSV
panmvpa/cli.py           stage runner
scripts/fetch_hub.py     S3 streaming download, one subject at a time
scripts/run_all.py       entry point
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
