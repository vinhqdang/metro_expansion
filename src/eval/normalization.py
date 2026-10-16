"""Anchor-based normalisation and augmented Tchebycheff scalarisation for (D_tot, GGI2, A_new)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

OBJECTIVE_KEYS: tuple[str, str, str] = ("D_tot", "ggi2", "A_new")
CLIP_MAX = 1.2


@dataclass
class Norm:
    """Reference anchors f_ref (one per objective) and Tchebycheff parameters z (ideal) and rho."""

    f_ref: np.ndarray = field(default_factory=lambda: np.ones(3))
    z: float = 1.05
    rho: float = 0.05

    def __post_init__(self) -> None:
        self.f_ref = np.asarray(self.f_ref, dtype=float).reshape(3)

    def normalize(self, obj: Mapping[str, Any] | Sequence[float] | np.ndarray) -> np.ndarray:
        """Divide by f_ref and clip to [0, 1.2]; accepts a dict with OBJECTIVE_KEYS or a length-3 array (or (n,3))."""
        if isinstance(obj, Mapping):
            raw = np.array([obj[k] for k in OBJECTIVE_KEYS], dtype=float)
        else:
            raw = np.asarray(obj, dtype=float)
        safe = np.where(self.f_ref > 0, self.f_ref, 1.0)
        scaled = np.where(self.f_ref > 0, raw / safe, 0.0)
        return np.clip(scaled, 0.0, CLIP_MAX)

    def g(self, f_norm: np.ndarray, w: Sequence[float] | np.ndarray) -> np.ndarray | float:
        """Augmented Tchebycheff g(f|w) = max_j w_j(z-f_j) + rho * sum_j w_j(z-f_j); works on (3,) or (n,3)."""
        f = np.asarray(f_norm, dtype=float)
        wv = np.asarray(w, dtype=float)
        d = wv * (self.z - f)
        val = d.max(axis=-1) + self.rho * d.sum(axis=-1)
        return float(val) if val.ndim == 0 else val


def save_norm(norm: Norm, path: str | Path) -> None:
    """Write a Norm to JSON."""
    payload = {
        "objectives": list(OBJECTIVE_KEYS),
        "f_ref": [float(x) for x in norm.f_ref],
        "z": float(norm.z),
        "rho": float(norm.rho),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def load_norm(path: str | Path) -> Norm:
    """Read a Norm from JSON written by save_norm."""
    payload = json.loads(Path(path).read_text())
    return Norm(f_ref=np.asarray(payload["f_ref"], dtype=float), z=float(payload["z"]), rho=float(payload["rho"]))
