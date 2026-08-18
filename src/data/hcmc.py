"""Ho Chi Minh City data loading and candidate-region graph construction.

Builds the GNN encoder's inputs (node features, spatial-contiguity edges,
OD-flow edges) from the real CSL_HCMC dataset (CityScope / MIT Media Lab,
github.com/CityScope/CSL_HCMC -- see docs/data_sources.md for provenance and
license notes). The repository is not vendored here; callers pass the path
to a local clone via `data_dir`.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def load_zones(data_dir: Path | str) -> gpd.GeoDataFrame:
    """Load HCMC's commune/ward-level candidate regions with population,
    income, and employment features (outputs/zones.geojson).

    These 322 zones are used directly as MNEP candidate expansion regions --
    they already carry the demand-relevant features (population density,
    income brackets, employment by sector) that MetroGNN-style pipelines
    otherwise have to construct from scratch.
    """
    path = Path(data_dir) / "outputs" / "zones.geojson"
    zones = gpd.read_file(path)
    return zones.to_crs(epsg=4326)


def load_bus_stops(data_dir: Path | str) -> gpd.GeoDataFrame:
    """Load existing bus stops -- the auxiliary demand-proxy/candidate-region
    signal (never an action target; see manuscript §Case Studies and Data)."""
    path = Path(data_dir) / "Data" / "GIS" / "Bus" / "Bus_shapefile" / "BusStop.shp"
    stops = gpd.read_file(path)
    return stops.to_crs(epsg=4326)


def load_od_survey(data_dir: Path | str) -> gpd.GeoDataFrame:
    """Load the 2018 household-travel-survey OD desire lines. `Start`/`End`
    are district-name strings (coarser than the 322-zone granularity), and
    `Quantity_M` is the surveyed OD flow magnitude."""
    path = Path(data_dir) / "Data" / "GIS" / "OD_2018" / "OD_shapefile" / "Survey_OD.shp"
    od = gpd.read_file(path)
    return od.to_crs(epsg=4326)


def build_node_features(zones: gpd.GeoDataFrame) -> np.ndarray:
    """Extract per-zone node features for the GNN encoder: population
    density (2019), total residential population by income bracket, and
    total employment. All min-max normalized to [0, 1] so they combine
    sensibly with other feature sources (e.g. bus-stop density).
    """
    income_cols = ["res_income_1", "res_income_2", "res_income_3", "res_income_4"]
    raw = zones[["Den_2019", *income_cols, "emp_total"]].fillna(0.0).to_numpy(dtype=np.float64)
    return _min_max_normalize(raw)


def _min_max_normalize(x: np.ndarray) -> np.ndarray:
    col_min = x.min(axis=0, keepdims=True)
    col_max = x.max(axis=0, keepdims=True)
    span = np.where(col_max > col_min, col_max - col_min, 1.0)
    return (x - col_min) / span


def build_bus_stop_density_feature(zones: gpd.GeoDataFrame, bus_stops: gpd.GeoDataFrame) -> np.ndarray:
    """Auxiliary bus-network signal: number of bus stops per zone, normalized
    by zone area (stops / km^2), min-max normalized to [0, 1].

    This is the concrete implementation of the "bus network as auxiliary
    signal" design decision: it becomes one additional node feature column,
    never a separate action space.
    """
    zones_projected = zones.to_crs(zones.estimate_utm_crs())
    stops_projected = bus_stops.to_crs(zones_projected.crs)

    joined = gpd.sjoin(stops_projected, zones_projected, how="inner", predicate="within")
    counts = joined.groupby("index_right").size()
    counts = counts.reindex(range(len(zones_projected)), fill_value=0)

    area_km2 = zones_projected.geometry.area / 1e6
    density = counts.to_numpy(dtype=np.float64) / np.maximum(area_km2.to_numpy(), 1e-6)
    return _min_max_normalize(density.reshape(-1, 1)).flatten()


def build_contiguity_edges(zones: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build spatial-contiguity edges (queen contiguity: zones sharing any
    boundary point) via the zones' spatial index. Returns (edge_index,
    edge_weight) with edge_index shape [2, num_edges] and unit weights,
    directly consumable by src/gnn/encoder.py. Edges are added in both
    directions (undirected graph).
    """
    sindex = zones.sindex
    src, dst = [], []
    for i, geom in enumerate(zones.geometry):
        candidate_idx = list(sindex.intersection(geom.bounds))
        for j in candidate_idx:
            if j <= i:
                continue
            if geom.touches(zones.geometry.iloc[j]) or geom.intersects(zones.geometry.iloc[j]):
                src.extend([i, j])
                dst.extend([j, i])

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float64)
    return edge_index, edge_weight


def build_od_edges(
    zones: gpd.GeoDataFrame, od: gpd.GeoDataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate district-level OD survey flows down to zone-level edges.

    The survey's `Start`/`End` are district names; each surveyed flow is
    distributed across all zones in the matched district(s), weighted by
    each zone's share of that district's 2019 population -- a standard
    disaggregation approach when OD data is coarser than the candidate-region
    granularity.
    """
    zone_dist = zones["Dist_Name"].to_numpy()
    zone_pop = zones["Pop_2019"].fillna(0.0).to_numpy(dtype=np.float64)

    dist_pop_total: dict[str, float] = {}
    for dist_name, pop in zip(zone_dist, zone_pop):
        dist_pop_total[dist_name] = dist_pop_total.get(dist_name, 0.0) + pop

    src, dst, weight = [], [], []
    for _, row in od.iterrows():
        start_zones = np.where(zone_dist == row["Start"])[0]
        end_zones = np.where(zone_dist == row["End"])[0]
        if len(start_zones) == 0 or len(end_zones) == 0:
            continue  # district name in survey has no matching zone (data-quality gap; log upstream)

        quantity = float(row["Quantity_M"]) if pd.notna(row["Quantity_M"]) else 0.0
        for i in start_zones:
            share_i = zone_pop[i] / max(dist_pop_total[zone_dist[i]], 1e-6)
            for j in end_zones:
                share_j = zone_pop[j] / max(dist_pop_total[zone_dist[j]], 1e-6)
                w = quantity * share_i * share_j
                if w > 0:
                    src.append(i)
                    dst.append(j)
                    weight.append(w)

    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float64)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = np.array(weight, dtype=np.float64)
    edge_weight = edge_weight / edge_weight.max()  # normalize to [0, 1] for comparability across cities
    return edge_index, edge_weight
