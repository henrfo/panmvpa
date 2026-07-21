"""Download template parcellations (the 'template' condition for the MVPA comparison).

Atlases are cached under ATLAS_DIR (henrik_sidequest/atlases/) via nilearn's fetchers,
so a clone + one download makes them available offline.
"""
from __future__ import annotations

from nilearn import datasets

from . import config


def fetch_schaefer(n_rois: int = 400, yeo_networks: int = 17, resolution_mm: int = 2):
    """Schaefer 2018 cortical parcellation (default 400 parcels, 17 networks, 2mm)."""
    return datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
        data_dir=str(config.ATLAS_DIR),
    )


def fetch_yeo(n_networks: int = 17, thickness: str = "thick"):
    """Yeo 2011 volumetric atlas (default 17 networks, thick). Returns Bunch with .maps."""
    return datasets.fetch_atlas_yeo_2011(
        n_networks=n_networks,
        thickness=thickness,
        data_dir=str(config.ATLAS_DIR),
    )


def download_all() -> dict:
    """Fetch every template atlas we need. Returns {name: maps_path}."""
    config.ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    schaefer = fetch_schaefer()
    yeo = fetch_yeo()
    return {
        "schaefer400_17": schaefer.maps,
        "yeo17_thick": yeo.maps,
    }
