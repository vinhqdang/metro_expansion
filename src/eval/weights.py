"""Simplex-lattice weight vectors for Tchebycheff scalarisation."""

from __future__ import annotations

from itertools import product

import numpy as np


def lattice(H: int = 4, m: int = 3) -> np.ndarray:
    """All m-dimensional weight vectors (i_1..i_m)/H with non-negative integers summing to H."""
    rows = [
        tuple(c)
        for c in product(range(H + 1), repeat=m)
        if sum(c) == H
    ]
    rows.sort(reverse=True)
    return np.asarray(rows, dtype=float) / H
