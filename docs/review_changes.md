# Changes made in response to the previous reviews

Reviews of the earlier submission (Transportmetrica B, decision of 15 October 2026), point by point. "Done" means the repository and the manuscript contain it; "Not done" says why.

## Reviewer 1
| # | Point | Status |
|---|---|---|
| 1 | Claims exceed the evidence; moderate title and contribution claims, or strengthen the evidence | Done. The title and abstract now state the comparison and its outcome: the learned policy is behind NSGA-II at equal budget. The words "fair" and "transferable" are dropped from the title. |
| 2 | Chengdu / larger network | Not done. No origin-destination data for Chengdu are available. Amsterdam (1,645 cells, about twice the size of Xi'an) was added as the larger instance, and the limitation is stated. |
| 3 | Fairness: degenerate case, endogenous price groups, complementary measures | Done for the measures: generalised Gini welfare (zero when nothing is served), worst-served share, and a Gini index reported as undefined, never zero, with counts. Not done: robustness runs with other groupings (3 or 10 groups, distance rings). The endogeneity of price groups is stated as a limitation. |
| 4 | Accessibility proxy; put the real accessibility into the reward | Partly. The uniform coverage proxy is replaced by trip-end mass newly reached, used in the reward and the evaluation. The radiation model could not be computed on the Xi'an and Amsterdam grids (no population or coordinates), and Ho Chi Minh City was dropped from this version, so the radiation model is not in the reward. |
| 5 | Transfer: one pair, three seeds, no component ablation | Partly. Both directions between Xi'an and Amsterdam, five paired seeds, six conditions including a random frozen encoder. Only the reduced budget was run. No contrast can be significant with five pairs, which the text says. |
| 6 | One deterministic pipeline for all results | Done. Every table is generated from run records by scripts; all results come from one pipeline with fixed seeds. |
| 7 | Streamline; remove revision history | Done. The manuscript is rewritten around the final method and experiments. |

## Reviewer 2
| # | Point | Status |
|---|---|---|
| 1 | Limited novelty | Reframed. The contribution is the corrected scoring protocol and the matched-budget comparison, not a new method. |
| 2 | Oversimplified formulation (cost, capacity, phasing) | Stated as limitations. Not modelled. |
| 3 | No clear performance advantage | Reported as found, with exact tests: the learned policy is not better than NSGA-II. |
| 4 | Baselines not faithful; no multi-objective algorithms; no Pareto quality | Done. NSGA-II, scalarised GAs, hypervolume, expected utility and IGD+. The "MetroGNN-like" and "Zhang et al.-like" rows are renamed as ablations of our own policy. |
| 5 | Equity measure validity | Done for the measure (see R1.3). Price-group endogeneity is stated. |
| 6 | Accessibility not justified | See R1.4. |
| 7 | Ho Chi Minh City data uncertainty, sensitivity analysis | Not done. Ho Chi Minh City was left out of this version. |
| 8 | Transferability evidence | See R1.5. |
| 9 | Inconsistent numbers | Done. Numbers in the text come from generated macros. |

## Reviewer 3
| # | Point | Status |
|---|---|---|
| 1 | Language: long sentences, defensive statements | Done. The text was rewritten with short sentences and without revision-history narration. |
| 2 | Replay and Monte Carlo returns | Done. The released code never used prioritized replay or reward shaping; both claims are removed. Training is batched on-policy REINFORCE with a leave-one-out baseline and terminal returns, described exactly in the methods section and Algorithm 1. |
| 3 | Report each objective and Pareto trade-offs | Done: per-objective table at the vertex weights, hypervolume, expected utility, IGD+ and a Pareto figure. |
| 4 | Compare with combinatorial optimisation | Done: NSGA-II and scalarised genetic algorithms at matched budgets. |
| 5 | Xi'an Line 1 experiment | Done as a leave-one-line-out test with ten seeds and a random reference. The OD matrix is a single snapshot recorded with both lines operating, which is stated. |

## Findings that change the paper's message
- At equal budget NSGA-II gives better deployable trade-off sets than the learned policy on Xi'an and Amsterdam, and the gap in search quality grows on the larger grid.
- The trained policy returns almost the same line for different weight vectors.
- NSGA-II recovers the real Line 2 better than a random reference; the learned policy does not.
- Warm starting helped from Amsterdam to Xi'an in 5 of 5 seeds and did not help in the other direction; five pairs cannot establish significance.
