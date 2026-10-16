# Analysis plan (fixed before any evaluation seed is run)

Evaluation seeds are 0-9. Tuning seeds are 100-102 and are used for normalisation anchors and hyperparameter selection only.
Nothing in this file is changed after the first evaluation run; deviations are recorded in the manuscript.

## Setting
- Cities: Xi'an (29x29, 20 stations, chained reward) and Amsterdam (35x47, 20 stations, chained reward).
- Objectives: demand share D, generalised Gini welfare GGI (base 2, five price groups), newly reached trip-end mass A.
- Scaling: anchors from single-objective GA (tuning seeds 100-102, 10,100 lines each); z = 1.05, rho = 0.05.
- Weight set: simplex lattice H = 4 (15 vectors).
- Budget: about 10,000 simulated lines per method (RL 10,000 episodes; NSGA-II 100 + 100x100; matched scalar GA 15 x (30 + 21x30); tabular 10,000 episodes; GA of Michailidis et al. with their default population 100, 50 generations).

## Hyperparameter selection
- RL: six configurations on tuning seeds 100-102 at the full budget, selected by mean expected utility (EUM) of the 15-point greedy front.
- NSGA-II: six population/generation splits at about the same total budget ((25,403), (50,201), (100,100), (200,49), (400,24), (800,12)), selected by mean EUM on the same tuning seeds. The matched scalar GA keeps a fixed 30 x 21 split per weight vector and is not tuned; the full-budget variant is reported as an upper reference.
- The selected configuration is then used unchanged for all evaluation runs.

## Primary endpoints (Xi'an, full network, chained, 20 stations)
1. Hypervolume of the search archive (Pareto set of all lines evaluated), RL full versus NSGA-II, 10 seeds.
2. Expected utility (EUM) of the 15-point deployable front, RL full versus NSGA-II, 10 seeds.
Test: two-sided Mann-Whitney U with Vargha-Delaney A12 and a 95% bootstrap interval of the mean difference. Holm correction over the two endpoints. alpha = 0.05.

## Secondary endpoints
- Per-objective values (D, s_min, GGI, A) at the three vertex weights and the centroid weight, all methods.
- Amsterdam: the same two primary endpoints.
- Line-1-only holdout (Line 2 removed, 27 stations): F1 at 1-cell tolerance between the generated line and the real Line 2, compared with 10,000 random feasible 27-cell lines.
- Transfer (Xi'an and Amsterdam, both directions, paired by seed): area under the EUM learning curve versus scratch, six conditions, Wilcoxon signed-rank with Holm correction within each (direction, budget) family.
- Ablations: demand-only reward, no message passing, no lookahead features, no start head, Gini reward, unscaled objectives.

## Claim rules
| Outcome (RL full against the better of NSGA-II and matched scalar GA, Holm-adjusted) | Claim allowed |
|---|---|
| RL better on both endpoints, p < 0.05, A12 >= 0.71 | "At a matched budget the learned policy finds better trade-off sets than NSGA-II on this benchmark." |
| No significant difference | "Comparable to NSGA-II at matched budget"; the remaining argument is that one trained policy answers any weight vector without a new search. |
| RL worse | Stated in the abstract. The contribution is the corrected evaluation protocol and the matched-budget comparison. |
- Transfer: "warm start helps" only for (direction, budget, condition) cells with Holm-adjusted p < 0.05. If the random frozen encoder performs like the transferred frozen encoder, the effect is attributed to fewer trainable parameters.
- Holdout: agreement with the real Line 2 is claimed only if F1 exceeds the 95th percentile of the random null, and is worded as "consistent with".
- Title and abstract do not use "fair" or "transferable" unless the fairness metrics and the transfer contrasts come out positive.

## Reporting rules
- Every number in the text comes from generated files (`manuscript/generated/numbers.tex`).
- Undefined fairness values (Gini or min/max ratio when nothing is served) are excluded and counted, never set to zero.
- Methods that return a single line (tabular Q-learning, greedy) are reported as single points.
- All failed or abandoned configurations are listed in the appendix.
