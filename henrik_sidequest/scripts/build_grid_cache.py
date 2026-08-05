"""Build the analysis-domain / group-map cache from the atlas alone — no BOLD needed.

The later stages (`compare`, `identify`, `figure`) only read `.npy` maps, but they used
to derive the voxel grid from a BOLD header. After `--cleanup` there is no BOLD left, so
they crashed. This writes the grid once so those stages never touch BOLD again.

    python scripts/build_grid_cache.py

Writes `derivatives/domain.npy` (3 x n_voxels coordinates) and
`derivatives/group_map.npy` (n_voxels network ids). Safe to re-run.

The grid geometry is hardcoded in config (MNI152NLin6Asym 2mm, 91x109x91), which is what
every ds006598 preproc BOLD carries and what the stored maps were built against. Pass
`--verify-against-maps` to confirm the rebuilt domain matches the maps already on disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panmvpa import config, parcellation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify-against-maps", action="store_true",
                    help="check the domain size matches the stored maps' length")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    if args.force:
        for p in (config.DOMAIN_CACHE, config.GROUP_MAP_CACHE):
            p.unlink(missing_ok=True)
        parcellation.analysis_domain.cache_clear()
        parcellation._domain_group_labels.cache_clear()
        parcellation.group_networks.cache_clear()

    shape, affine = parcellation.grid()
    print(f"grid shape : {shape}")
    print(f"grid affine:\n{affine}")
    print(f"atlas      : {config.ATLAS_IMAGE.name}")
    if not config.ATLAS_IMAGE.exists():
        raise SystemExit(f"Atlas not found at {config.ATLAS_IMAGE}")

    domain_path, group_path = parcellation.write_grid_cache()
    idx = np.load(domain_path)
    labels = np.load(group_path)
    print(f"\ndomain     -> {domain_path}  {idx.shape} ({idx.shape[1]:,} voxels)")
    print(f"group map  -> {group_path}  {labels.shape}")
    print(f"networks   : {sorted(set(labels.tolist()))}")

    if args.verify_against_maps:
        stored = sorted(config.MAPS_DIR.glob("*.npy")) if config.MAPS_DIR.exists() else []
        if not stored:
            print(f"\n!! no stored maps found in {config.MAPS_DIR} to verify against")
            raise SystemExit(1)
        lengths = {p.name: int(np.load(p).shape[0]) for p in stored}
        distinct = set(lengths.values())
        print(f"\nverifying against {len(stored)} stored maps")
        print(f"  stored map lengths : {sorted(distinct)}")
        print(f"  rebuilt domain size: {idx.shape[1]}")
        if distinct != {idx.shape[1]}:
            print("  MISMATCH -- the cache does NOT match the existing maps.")
            for name, n in sorted(lengths.items())[:5]:
                print(f"    {name}: {n}")
            raise SystemExit(2)
        print("  MATCH -- existing maps are usable with this cache.")


if __name__ == "__main__":
    main()
