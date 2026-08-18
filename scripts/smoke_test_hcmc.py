"""Manual integration smoke test against the real CSL_HCMC data.

Not part of the pytest suite (external/ is gitignored and not guaranteed to
be present). Run after cloning github.com/CityScope/CSL_HCMC into
external/CSL_HCMC:

    python scripts/smoke_test_hcmc.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hcmc import (  # noqa: E402
    build_bus_stop_density_feature,
    build_contiguity_edges,
    build_node_features,
    build_od_edges,
    load_bus_stops,
    load_od_survey,
    load_zones,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "external" / "CSL_HCMC"


def main() -> None:
    if not DATA_DIR.exists():
        print(f"CSL_HCMC not found at {DATA_DIR} -- clone it first:")
        print("  git clone https://github.com/CityScope/CSL_HCMC.git external/CSL_HCMC")
        sys.exit(1)

    t0 = time.time()
    zones = load_zones(DATA_DIR)
    bus_stops = load_bus_stops(DATA_DIR)
    od = load_od_survey(DATA_DIR)
    print(f"loaded {len(zones)} zones, {len(bus_stops)} bus stops, {len(od)} OD desire lines "
          f"in {time.time() - t0:.2f}s")

    node_features = build_node_features(zones)
    print(f"node_features: shape={node_features.shape}, "
          f"range=[{node_features.min():.3f}, {node_features.max():.3f}]")

    bus_density = build_bus_stop_density_feature(zones, bus_stops)
    print(f"bus_stop_density: shape={bus_density.shape}, "
          f"nonzero_zones={int((bus_density > 0).sum())}/{len(bus_density)}")

    t1 = time.time()
    spatial_edge_index, spatial_edge_weight = build_contiguity_edges(zones)
    print(f"contiguity edges: {spatial_edge_index.shape[1]} directed edges "
          f"in {time.time() - t1:.2f}s")

    t2 = time.time()
    od_edge_index, od_edge_weight = build_od_edges(zones, od)
    print(f"OD edges: {od_edge_index.shape[1]} directed edges "
          f"(from {len(od)} district-level survey rows) in {time.time() - t2:.2f}s")
    if od_edge_index.shape[1] == 0:
        print("WARNING: zero OD edges produced -- check district name matching "
              "between Survey_OD.Start/End and zones.Dist_Name")

    print(f"\ntotal: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
