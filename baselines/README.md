# Baselines

This directory was originally scaffolded for from-scratch reimplementations of the three
reference RL methods (`metrognn/`, `zhang2024_multiobjective/`, `michailidis_tabular/`).
None of these subdirectories were ever populated, and the manuscript does not use this
route for any of its actual comparisons:

- MetroGNN (su2024) and Zhang et al. (zhang2024) have no public code release, so the
  manuscript instead uses two principled ablations of its own architecture
  (`--ablation single_objective` / `--ablation flat_encoder` in `scripts/train_xian.py`
  and `scripts/train_hcmc.py`) as stand-ins -- see the manuscript's Baselines subsection
  for exactly what each ablation changes and why it approximates, not reproduces, the
  corresponding paper.
- Michailidis et al.'s tabular RL and genetic-algorithm baselines both have public code,
  so the manuscript runs their own code release directly and unmodified (aside from one
  documented upstream bug fix and one instrumentation addition) rather than
  reimplementing either -- see `docs/baseline_reproduction.md` for exact setup, the
  patches applied, and reproduction commands.

This directory's three empty subdirectories are accordingly a stale, unused scaffold
from an earlier planning pass, not work in progress; see
`docs/superpowers/specs/2026-08-18-metro-expansion-manuscript-design.md` §5 for the
original design rationale if reviving them is ever useful.
