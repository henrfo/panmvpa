"""Step 5 smoke test — prove the plumbing works end to end.

Discovers a subject's epiproj sessions, builds the events table, loads one BOLD run,
and reports the template atlases. Run after cloning + `datalad get` for one subject:

    python verify.py --subject PAN01
"""
from __future__ import annotations

import argparse

import panmvpa
from panmvpa import atlases, config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="PAN01")
    ap.add_argument("--skip-atlases", action="store_true")
    args = ap.parse_args()

    sub = config.sub_id(args.subject)
    print(f"=== {sub} ===")
    print(f"data root : {config.DATA_ROOT}")
    print(f"TR        : {config.TR}s | event duration: {config.EVENT_DURATION}s")

    sessions = panmvpa.epiproj_sessions(sub)
    print(f"epiproj sessions: {sessions}")
    if not sessions:
        raise SystemExit("No epiproj sessions found — clone the dataset first.")

    ses = sessions[0]
    events = panmvpa.build_events(sub, ses)
    print(f"\n--- events: {sub} ses-{ses} ({len(events)} rows) ---")
    print(events.to_string(index=False))
    print("\ntrials per condition:")
    print(events.trial_type.value_counts().to_string())

    if panmvpa.is_fetched(sub, ses):
        img = panmvpa.load_bold(sub, ses)
        print(f"\nBOLD shape: {img.shape} | voxel size: {img.header.get_zooms()}")
    else:
        print(f"\nBOLD not fetched yet: python fetch_data.py --subject {sub} --sessions {ses}")

    if not args.skip_atlases:
        print("\n--- template atlases ---")
        for name, path in atlases.download_all().items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
