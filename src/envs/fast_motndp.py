"""Gather-based MOTNDP environment, numerically identical to the upstream env.

The upstream MOTNDP._calculate_reward builds a dense (grid_size x grid_size)
mask of satisfied OD pairs and multiplies it with each group OD matrix. This
module keeps the upstream pair construction (chained reward, connections to
existing lines, segments already covered) and replaces only the dense
multiply with a gather over the satisfied pairs:

    sat_g = group_od_mx[g, i, j] summed over the unique satisfied pairs (i, j)

Duplicate pairs are removed before the gather, because the dense mask counts
each (i, j) at most once. Excluded OD segments need no handling here: the
City constructor already zeroes them in od_mx and group_od_mx.

info["segments"] is provided by the upstream _get_info, so it is not changed.

Upstream code under external/ is only imported, never modified.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from gymnasium.envs.registration import register, registry

REPO_ROOT = Path(__file__).resolve().parents[2]
MOTNDP_PKG_DIR = REPO_ROOT / "external" / "tabular-tndp" / "envs" / "mo-tndp"
if str(MOTNDP_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(MOTNDP_PKG_DIR))

from motndp.motndp import MOTNDP  # noqa: E402
from motndp.city import City  # noqa: E402

ENV_ID = "motndp_fast-v0"
CITIES_ROOT = MOTNDP_PKG_DIR / "cities"
DATA_CITIES_ROOT = REPO_ROOT / "data" / "cities"
CITY_NAMES = ("xian", "amsterdam")
CITY_VARIANTS = ("full", "line1_only", "line2_only")


class FastMOTNDP(MOTNDP):
    """MOTNDP with a gather-based _calculate_reward. Everything else is inherited."""

    def _satisfied_pairs(self, segment, cells_to_chain, connected_cells):
        """Satisfied (i, j) pairs, same construction as City.satisfied_od_mask."""
        city = self.city
        # Pairs within the new segment (a single pair for a 2-cell step).
        sat_od_pairs = np.array(list(itertools.combinations(segment.flatten(), 2)))

        if cells_to_chain is not None and len(cells_to_chain) > 0:
            cells_to_chain = cells_to_chain[cells_to_chain != segment[0]]
            sat_od_pairs = np.concatenate(
                (sat_od_pairs, np.column_stack((cells_to_chain, np.full(len(cells_to_chain), segment[1]))))
            )

        if connected_cells is not None and len(connected_cells) > 0:
            connected_cells = np.asarray(list(connected_cells))
            filtered_segment = np.setdiff1d(segment, np.concatenate(city.existing_lines), assume_unique=True)
            new_line_cells = np.hstack((filtered_segment, cells_to_chain))
            new_sat_od_pairs = np.array(
                np.stack(np.meshgrid(new_line_cells, connected_cells), axis=-1).reshape(-1, 2)
            )
            sat_od_pairs = np.vstack((sat_od_pairs, new_sat_od_pairs))

        return sat_od_pairs

    def _calculate_reward(self, segment):
        assert (
            self.city.group_od_mx is not None
        ), "Cannot use multi-objective reward without group definitions. Provide --groups_file argument"

        n_groups = self.city.group_od_mx.shape[0]
        # Same early return as upstream: an already covered segment earns nothing.
        if segment in self.covered_segments:
            return np.zeros(n_groups), np.empty((0, 2))

        segment = np.asarray(segment)
        if self.chained_reward:
            cells_to_chain = np.asarray(
                self.city.grid_to_index(np.array(self.covered_cells_coordinates))
            )
            connected_stations = self.city.connections_with_existing_lines(segment)
            if connected_stations:
                self.connections_with_existing_lines.update(connected_stations)
            sat_od_pairs = self._satisfied_pairs(segment, cells_to_chain, self.connections_with_existing_lines)
        else:
            sat_od_pairs = self._satisfied_pairs(segment, None, None)

        # Drop pairs already covered (upstream: segments_to_ignore=self.covered_segments).
        if len(self.covered_segments) > 0:
            ignore_set = set(map(tuple, self.covered_segments))
            sat_od_pairs = np.array(
                [pair for pair in sat_od_pairs if tuple(pair) not in ignore_set]
            ).reshape(-1, 2)

        # Gather group ODs over the unique satisfied pairs (the dense mask counts each pair once).
        if sat_od_pairs.shape[0] > 0:
            uniq = np.unique(sat_od_pairs, axis=0)
            sat_group_ods = self.city.group_od_mx[:, uniq[:, 0], uniq[:, 1]].sum(axis=1)
        else:
            sat_group_ods = np.zeros(n_groups)

        if self.od_type == "pct":
            sat_group_ods_pct = np.divide(
                sat_group_ods,
                self.city.group_od_sum,
                out=np.zeros_like(sat_group_ods),
                where=self.city.group_od_sum != 0,
            )
            group_rw = sat_group_ods_pct
        else:
            group_rw = sat_group_ods

        return group_rw, sat_od_pairs


def register_fast_envs() -> None:
    """Register ENV_ID with gymnasium. Safe to call more than once."""
    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point="src.envs.fast_motndp:FastMOTNDP")


def _load_variant_generator():
    import importlib.util

    path = REPO_ROOT / "scripts" / "make_city_variants.py"
    spec = importlib.util.spec_from_file_location("make_city_variants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_city(name: str, variant: str = "full", groups_file: str = "price_groups_5.txt") -> City:
    """Load a city. variant 'full' is the upstream folder; other variants live in data/cities/."""
    if name not in CITY_NAMES:
        raise ValueError(f"unknown city {name!r}; expected one of {CITY_NAMES}")
    if variant not in CITY_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {CITY_VARIANTS}")

    if variant == "full":
        city_dir = CITIES_ROOT / name
    else:
        city_dir = DATA_CITIES_ROOT / f"{name}_{variant}"
        if not (city_dir / "config.txt").exists():
            _load_variant_generator().make_variant(name, variant, DATA_CITIES_ROOT)
    return City(city_dir, groups_file=groups_file)


def make_env(city: City, nr_stations: int, chained: bool):
    """Build the fast env through the gymnasium registry (od_type 'pct', MetroConstraints)."""
    import mo_gymnasium as mo_gym

    from motndp.constraints import MetroConstraints

    register_fast_envs()
    return mo_gym.make(
        ENV_ID,
        city=city,
        constraints=MetroConstraints(city),
        nr_stations=nr_stations,
        od_type="pct",
        chained_reward=chained,
    )


def interpolate_path(cells):
    """Fill gaps between consecutive cells with Chebyshev steps.

    Each step moves one cell in x and/or y towards the next station, so a jump
    such as (8,2)->(10,3) becomes (8,2)->(9,3)->(10,3). Cells that are already
    adjacent are kept as they are.
    """
    cells = [tuple(int(v) for v in c) for c in cells]
    if not cells:
        return []
    out = [cells[0]]
    for nxt in cells[1:]:
        cur = out[-1]
        while cur != nxt:
            dx = nxt[0] - cur[0]
            dy = nxt[1] - cur[1]
            cur = (cur[0] + int(np.sign(dx)), cur[1] + int(np.sign(dy)))
            out.append(cur)
    return [list(c) for c in out]
