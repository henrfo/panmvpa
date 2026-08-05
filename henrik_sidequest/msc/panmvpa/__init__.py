"""panmvpa (MSC fork) — a frozen copy of the FC sidequest, retargeted at the Midnight
Scan Club (OpenNeuro ds000224) to push the reliability question past 90 minutes.

Deliberately minimal: the FC pipeline (scripts/run_fc.py) reaches the dataset only through
`config`, `rest`, and `plotstyle`, so those are the only modules copied here. The legacy
WTA-maps modules (figure/identify/parcellation) are NOT part of this fork and are not imported.

Import submodules directly:

    from panmvpa import config, rest
    from panmvpa import plotstyle as ps
"""
from __future__ import annotations
