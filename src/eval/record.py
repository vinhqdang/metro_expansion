"""Shared run-record helpers so every method writes the same JSON schema (mnep.run.v1)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.eval.normalization import Norm
from src.eval.objectives import CityContext, compute_objectives
from src.eval.pareto import hypervolume, nondominated, eum
from src.eval.weights import lattice

SCHEMA = "mnep.run.v1"


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def segments_to_pairs(segments: Any) -> list[tuple[int, int]]:
    """env info['segments'] (list of [i, j]) to a list of int pairs."""
    arr = np.asarray(segments).reshape(-1, 2)
    return [(int(a), int(b)) for a, b in arr]


def make_point(
    ctx: CityContext,
    norm: Norm,
    w: Sequence[float] | None,
    mode: str,
    cells: Sequence[Sequence[int]],
    segments: Any,
    s: np.ndarray,
) -> dict[str, Any]:
    """One evaluated line: raw objectives, normalised vector and (if w is given) g_w."""
    cells_t = [tuple(int(v) for v in c) for c in cells]
    obj = compute_objectives(ctx, cells_t, segments_to_pairs(segments), s)
    obj["s"] = np.asarray(s, dtype=float).tolist()
    f_norm = norm.normalize(obj)
    point = {
        "w": None if w is None else [float(v) for v in w],
        "mode": mode,
        "start": list(cells_t[0]) if cells_t else None,
        "cells": [list(c) for c in cells_t],
        "obj": obj,
        "f_norm": f_norm.tolist(),
        "g_w": None if w is None else float(norm.g(f_norm, w)),
    }
    return _clean(point)


def front_indicators(f_norm: np.ndarray, norm: Norm, W: np.ndarray | None = None) -> dict[str, float]:
    """Hypervolume (ref = origin) and expected utility over the 15-vector lattice for a set of normalised points."""
    F = np.asarray(f_norm, dtype=float).reshape(-1, 3)
    W = lattice(4, 3) if W is None else W
    nd = F[nondominated(F)]
    return {
        "hv": float(hypervolume(nd, ref=(0.0, 0.0, 0.0))),
        "eum": float(eum(F, W, lambda P, w: norm.g(P, w))),
        "n_nondominated": int(nd.shape[0]),
    }


def save_record(path: str | Path, record: dict[str, Any]) -> None:
    record = dict(record)
    record["schema"] = SCHEMA
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(record), indent=1))
