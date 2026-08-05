# MSC fork of the FC sidequest

A **frozen copy** of the ds006598 FC pipeline, retargeted at the **Midnight Scan Club**
(OpenNeuro `ds000224`: 10 subjects × ~5 h rest each, one scanner). The point is to push the
reliability question **past 90 minutes** while keeping a real cohort, so both halves of the
analysis still work — within-person convergence *and* between-person individuation — with no
N=1 / cross-scanner batch confound.

This directory is intentionally **decoupled** from the parent sidequest: nothing here imports
from `../panmvpa`, and because `config.py` derives all paths from its own `__file__`, every
output lands under `msc/derivatives/` and `msc/results/` — it cannot collide with the ds006598
run. Fix a bug here and it will **not** flow back to the original, and vice versa. That is the
accepted trade for freezing the known-good pipeline.

## Layout (only what run_fc.py actually needs)

```
msc/
  panmvpa/__init__.py     minimal — no legacy WTA imports
  panmvpa/config.py       COPY — edit for MSC (subjects, TR, space, task, atlas)
  panmvpa/rest.py         COPY — edit rest_runs() for the ds000224 file layout
  panmvpa/plotstyle.py    COPY, frozen — shared style, do not diverge
  scripts/run_fc.py       COPY — analysis code; retune the N and minute-grid for MSC depth
```

## How to run (from this folder, so `panmvpa` resolves to the fork)

```bash
cd msc
PYTHONPATH=. DATA_DIR=/path/to/ds000224 python scripts/run_fc.py reduce
PYTHONPATH=. python scripts/run_fc.py analyze
```

As copied, these files are **byte-identical to the ds006598 pipeline** — point `DATA_DIR` at
ds006598 and it reproduces the original result, a sanity check that the fork is wired before we
diverge.

## Adapter TODO (the real work — pending a look at what ds000224 ships)

1. **`config.py`** — `SUBJECTS` (MSC01–10), `TR` (MSC's, not 1.355), `SPACE`, `REST_TASK`,
   and whether the Schaefer-400 atlas needs a different space/regrid for MSC's preproc grid.
2. **`rest.py` → `rest_runs()`** — the one BIDS-discovery function; rewrite for the ds000224
   filenames/session layout. Everything downstream consumes its output unchanged.
3. **`scripts/run_fc.py`** — extend the minute grid past 90 (MSC reaches ~300 min/subject);
   the within-person and sampling logic already scale, but the rungs/`emerge` columns are
   tuned for ~90 min and should be widened.
4. **Data** — decide MSC preproc BOLD (run our Schaefer masker → same raw-400 format) vs. a
   released parcellated derivative (smaller, but inherits their cleaning; less apples-to-apples
   with ds006598).

See `../` for the parent pipeline this was forked from.
