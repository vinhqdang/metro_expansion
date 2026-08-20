import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry import box

from src.data.hcmc import (
    _min_max_normalize,
    _normalize_district_name,
    build_bus_stop_density_feature,
    build_contiguity_edges,
    build_node_features,
    build_od_edges,
    build_radiation_probabilities,
    radiation_accessibility_score,
)


def make_toy_zones() -> gpd.GeoDataFrame:
    """A 2x2 grid of 1x1-degree squares: zone 0 and 1 are adjacent (share an
    edge), as are 0&2, 1&3, 2&3; 0&3 and 1&2 only touch at a corner."""
    geoms = [
        box(0, 0, 1, 1),  # zone 0
        box(1, 0, 2, 1),  # zone 1 (adjacent to 0)
        box(0, 1, 1, 2),  # zone 2 (adjacent to 0)
        box(1, 1, 2, 2),  # zone 3 (adjacent to 1 and 2)
    ]
    return gpd.GeoDataFrame(
        {
            "Dist_Name": ["A", "A", "B", "B"],
            "Pop_2019": [100.0, 300.0, 50.0, 150.0],
            "Den_2019": [10.0, 30.0, 5.0, 15.0],
            "res_income_1": [1.0, 2.0, 3.0, 4.0],
            "res_income_2": [1.0, 2.0, 3.0, 4.0],
            "res_income_3": [1.0, 2.0, 3.0, 4.0],
            "res_income_4": [1.0, 2.0, 3.0, 4.0],
            "emp_total": [10.0, 20.0, 30.0, 40.0],
            "geometry": geoms,
        },
        crs="EPSG:4326",
    )


def make_toy_bus_stops() -> gpd.GeoDataFrame:
    # 2 stops in zone 0, 1 stop in zone 1, 0 stops in zones 2/3
    points = [Point(0.5, 0.5), Point(0.2, 0.8), Point(1.5, 0.5)]
    return gpd.GeoDataFrame({"geometry": points}, crs="EPSG:4326")


def make_toy_od() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Start": ["A", "B"],
            "End": ["B", "A"],
            "Quantity_M": [100.0, 50.0],
        }
    )


def test_min_max_normalize_scales_to_unit_range():
    x = np.array([[0.0], [5.0], [10.0]])
    out = _min_max_normalize(x)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(0.5)


def test_min_max_normalize_handles_constant_column():
    x = np.array([[3.0], [3.0], [3.0]])
    out = _min_max_normalize(x)
    assert np.all(out == 0.0)  # span falls back to 1.0, avoiding div-by-zero


def test_build_node_features_shape_and_range():
    zones = make_toy_zones()
    features = build_node_features(zones)
    assert features.shape == (4, 6)  # Den_2019 + 4 income cols + emp_total
    assert features.min() >= 0.0
    assert features.max() <= 1.0


def test_build_contiguity_edges_connects_adjacent_zones():
    zones = make_toy_zones()
    edge_index, edge_weight = build_contiguity_edges(zones)

    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # 0-1, 0-2, 1-3, 2-3 should all be connected (share a full edge)
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        assert (a, b) in edges and (b, a) in edges
    assert np.all(edge_weight == 1.0)


def test_build_contiguity_edges_is_undirected():
    zones = make_toy_zones()
    edge_index, _ = build_contiguity_edges(zones)
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    for a, b in edges:
        assert (b, a) in edges


def test_build_bus_stop_density_feature_reflects_stop_counts():
    zones = make_toy_zones()
    bus_stops = make_toy_bus_stops()
    density = build_bus_stop_density_feature(zones, bus_stops)

    assert density.shape == (4,)
    # zone 0 has 2 stops in a 1x1-degree cell -> highest raw density -> normalized to 1.0
    assert density[0] == pytest.approx(1.0)
    # zones 2 and 3 have no stops -> zero density -> normalized to 0.0
    assert density[2] == pytest.approx(0.0)
    assert density[3] == pytest.approx(0.0)


def test_build_od_edges_distributes_district_flow_by_population_share():
    zones = make_toy_zones()
    od = make_toy_od()
    edge_index, edge_weight = build_od_edges(zones, od)

    assert edge_index.shape[1] > 0
    assert edge_weight.max() == pytest.approx(1.0)  # normalized

    # District A -> District B flow (100.0) should be split among zone-pairs
    # (0,1)x(2,3) weighted by population share within each district.
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_weight.tolist()))
    ab_edges = [(s, d, w) for s, d, w in edges if s in (0, 1) and d in (2, 3)]
    assert len(ab_edges) == 4  # 2 origin zones x 2 destination zones


def test_build_od_edges_skips_unmatched_district_names():
    zones = make_toy_zones()
    od = pd.DataFrame({"Start": ["NoSuchDistrict"], "End": ["A"], "Quantity_M": [10.0]})
    edge_index, edge_weight = build_od_edges(zones, od)
    assert edge_index.shape == (2, 0)
    assert edge_weight.shape == (0,)


def test_build_od_edges_handles_empty_od():
    zones = make_toy_zones()
    od = pd.DataFrame({"Start": [], "End": [], "Quantity_M": []})
    edge_index, edge_weight = build_od_edges(zones, od)
    assert edge_index.shape == (2, 0)


def test_normalize_district_name_strips_spaces_and_case():
    assert _normalize_district_name("Binh Tan") == _normalize_district_name("BinhTan")
    assert _normalize_district_name("Binh Tan") == "binhtan"


def test_build_od_edges_matches_district_names_despite_spacing_difference():
    # Regression test: CSL_HCMC's real survey data uses "BinhTan" (no space)
    # while the zones dataset uses "Binh Tan" (with space) for the same
    # district -- exact-string matching alone silently drops these rows.
    zones = make_toy_zones()
    zones.loc[0, "Dist_Name"] = "Binh Tan"
    od = pd.DataFrame({"Start": ["BinhTan"], "End": ["B"], "Quantity_M": [40.0]})

    edge_index, edge_weight = build_od_edges(zones, od)

    assert edge_index.shape[1] > 0  # previously this silently produced zero edges
    src_zones = set(edge_index[0].tolist())
    assert 0 in src_zones  # zone 0 ("Binh Tan") is correctly matched as an origin


def make_line_zones() -> gpd.GeoDataFrame:
    """Three zones on a line, A(0,0)-B(1,0)-C(3,0), with distinct pairwise
    distances (no ties) so the radiation model's intervening-population
    term s_ij is unambiguous to hand-verify."""
    geoms = [Point(0, 0).buffer(0.1), Point(1, 0).buffer(0.1), Point(3, 0).buffer(0.1)]
    return gpd.GeoDataFrame(
        {
            "Pop_2019": [100.0, 200.0, 50.0],
            "x_centroid": [0.0, 1.0, 3.0],
            "y_centroid": [0.0, 0.0, 0.0],
            "geometry": geoms,
        },
        crs="EPSG:4326",
    )


def test_build_radiation_probabilities_matches_hand_computed_values():
    zones = make_line_zones()
    p = build_radiation_probabilities(zones)

    assert p.shape == (3, 3)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert np.all(np.diag(p) == 0.0)  # self-flow excluded

    # A -> B: nothing lies strictly between them, s_AB = 0.
    assert p[0, 1] == pytest.approx((100 * 200) / ((100 + 0) * (100 + 200 + 0)))
    # A -> C: B (pop 200) lies strictly between, s_AC = 200.
    assert p[0, 2] == pytest.approx((100 * 50) / ((100 + 200) * (100 + 50 + 200)))
    # B -> A: nothing lies strictly between them, s_BA = 0.
    assert p[1, 0] == pytest.approx((200 * 100) / ((200 + 0) * (200 + 100 + 0)))
    # B -> C: A (pop 100) is closer to B (distance 1) than C is (distance 2), so it lies strictly
    # between them, s_BC = 100.
    assert p[1, 2] == pytest.approx((200 * 50) / ((200 + 100) * (200 + 50 + 100)))
    # C -> B: nothing lies strictly between them, s_CB = 0.
    assert p[2, 1] == pytest.approx((50 * 200) / ((50 + 0) * (50 + 200 + 0)))
    # C -> A: B (pop 200) lies strictly between, s_CA = 200.
    assert p[2, 0] == pytest.approx((50 * 100) / ((50 + 200) * (50 + 100 + 200)))


def test_build_radiation_probabilities_asymmetric():
    # The radiation model is not symmetric: p_ij != p_ji in general, since
    # the normalization depends on the ORIGIN's own population.
    zones = make_line_zones()
    p = build_radiation_probabilities(zones)
    assert p[0, 1] != pytest.approx(p[1, 0])


def test_radiation_accessibility_score_bounded_and_monotonic_in_stations():
    zones = make_line_zones()
    p = build_radiation_probabilities(zones)

    empty_score = radiation_accessibility_score(p, [])
    assert empty_score == 0.0

    one_station_score = radiation_accessibility_score(p, [1])
    all_stations_score = radiation_accessibility_score(p, [0, 1, 2])

    assert 0.0 <= one_station_score <= 1.0
    assert 0.0 <= all_stations_score <= 1.0
    # Adding more built stations can only raise (never lower) each region's
    # best-connection probability, so the aggregate score is monotonic.
    assert all_stations_score >= one_station_score


def test_build_od_edges_logs_warning_for_unmatched_rows_but_keeps_matched_ones(caplog):
    zones = make_toy_zones()
    od = pd.DataFrame(
        {
            "Start": ["A", "NoSuchDistrict"],
            "End": ["B", "A"],
            "Quantity_M": [100.0, 999.0],
        }
    )

    with caplog.at_level("WARNING"):
        edge_index, edge_weight = build_od_edges(zones, od)

    # the valid A->B row still produces edges...
    assert edge_index.shape[1] > 0
    # ...and the unmatched row is reported, not silently dropped.
    assert any("NoSuchDistrict" in record.message for record in caplog.records)
