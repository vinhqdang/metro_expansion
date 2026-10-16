"""Baseline methods on the shared genome and the upstream GA/Q-learning code (design sections 3.3-3.4).

evo.py holds NSGA-II, the single-weight GA, random lines and greedy-by-g. run.py holds the
drivers that return mnep.run.v1 records. upstream.py imports the unmodified upstream code.
Upstream imports are kept out of this package __init__ so that importing evo stays light.
"""

from src.baselines.evo import (
    Individual,
    Problem,
    crossover,
    crowding_distance,
    fast_nondominated_sort,
    ga_scalar,
    generational_ga,
    greedy_by_g,
    mutate,
    nsga2,
    random_lines,
)

__all__ = [
    "Individual",
    "Problem",
    "crossover",
    "crowding_distance",
    "fast_nondominated_sort",
    "ga_scalar",
    "generational_ga",
    "greedy_by_g",
    "mutate",
    "nsga2",
    "random_lines",
]
