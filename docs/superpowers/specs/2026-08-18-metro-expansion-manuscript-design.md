# Design: Fair and Transferable Graph RL for Metro Network Expansion

**Target venue:** *Railway Engineering Science* (Springer / Southwest Jiaotong University, open access, no APC)
**Status:** Approved for planning (2026-08-18)

## 1. Background & Gap

Reference literature (4 papers, all present in repo root as PDFs):

1. **Kang et al., Omega (2019)** — "Last train timetabling optimization and bus bridging service management in urban railway transit networks." MILP + decomposition, operational-phase problem on an *existing* network (last-train connections, bus bridging). Different sub-problem from the other three; used as related work / motivation for an operations-aware framing, not a direct baseline.
2. **Zhang et al., Transportation Research Part C (2024)** — "City metro network expansion based on multi-objective reinforcement learning." Formulates Metro Network Expansion Problem (MNEP) as sequential station selection (MDP), actor-critic, Tchebycheff decomposition over three objectives: OD demand, social equity, radiation accessibility. No GNN; validated on real-world data, beats heuristics by >30%.
3. **Su et al., MetroGNN, WWW'24 Companion (arXiv 2403.09197)** — GNN + attentive policy network over a heterogeneous multi-graph (spatial contiguity + OD-flow edges) for MNEP. Single-objective (OD demand satisfaction). >30% improvement over prior heuristics/RL on real urban data (Beijing-style).
4. **Michailidis, Ghebreab, Santos, arXiv 2606.04167 (2026)** — "Smart Transportation Without Neurons." Reformulates MNEP as a Non-Markovian Reward Decision Process to shrink the state-action space, argues Deep RL/GNN is unnecessary at MNEP's scale, uses tabular Monte Carlo RL with fairness-based reward functions. Validated on Xi'an and Amsterdam: 18x fewer training episodes, 12x less CO2 than deep RL baselines, comparable performance.

**Gap:** No paper combines GNN-based scalable state representation (MetroGNN) with multi-objective fairness decomposition (Zhang et al.) in a single policy. No paper tests whether Michailidis et al.'s "deep RL is unnecessary" claim holds at a genuinely large-scale network (e.g. Chengdu, ~500+ km) or on small, previously-unstudied emerging networks (Hanoi, Ho Chi Minh City). All three MNEP papers train and evaluate per-city from scratch; none attempt cross-city transfer.

## 2. Contribution

1. **GNN-based multi-objective policy**: extend MetroGNN's heterogeneous-graph attentive actor-critic with a Tchebycheff-decomposed vector reward (OD demand, social equity, radiation accessibility), following Zhang et al.'s objective formulation.
2. **Cross-city transfer/curriculum + sample-efficiency mechanism**: pretrain policy weights on smaller networks (Hanoi, Xi'an), fine-tune on larger ones (Ho Chi Minh City, Chengdu); combine with prioritized experience replay and potential-based reward shaping. Directly responds to Michailidis et al.'s carbon-cost critique without abandoning the GNN.
3. **First MNEP study spanning small-emerging to ultra-large-mature networks**: Hanoi → Ho Chi Minh City → Xi'an → Chengdu in one consistent experimental protocol, producing scale-dependent guidance (when does GNN/deep RL pay for itself vs. tabular RL) that single-city-scale prior work cannot offer.

## 3. Method

- **State encoder**: heterogeneous multi-graph GNN over urban regions/candidate stations. Two edge types: spatial contiguity, OD-flow association (per MetroGNN §3.1–3.2). Produces per-node embeddings via message passing.
- **Policy/value network**: attentive actor-critic over embedded graph; action = select next node (extend existing line / start new line), masked for network-design feasibility constraints (station spacing, angles, budget).
- **Reward**: vector reward `(demand_satisfaction, social_equity, radiation_accessibility)`, scalarized via Tchebycheff decomposition (Zhang et al. §3.2) to support Pareto-style exploration rather than a fixed weighted sum.
- **Efficiency layer**:
  - Cross-city curriculum: train on Hanoi and Xi'an first; initialize Ho Chi Minh City and Chengdu training from these weights.
  - Prioritized experience replay on transition batches.
  - Potential-based reward shaping to accelerate credit assignment (does not alter optimal policy).

## 4. Case Studies & Data

| City | Role | Scale |
|---|---|---|
| Hanoi | Novel, small/emerging | 2 operating lines + planned extensions |
| Ho Chi Minh City | Novel, medium/growing | 1 operating line (2024) + planned lines |
| Xi'an | Direct benchmark vs. Michailidis et al. | Medium, previously studied |
| Chengdu | Scale stress-test, journal-publisher relevance (SWJTU is Chengdu-based) | ~500+ km, one of world's largest networks |

**Open item (unresolved, needs research before data pipeline is finalized):** real OD/travel-demand data availability for Hanoi and Ho Chi Minh City. Fallback plan used elsewhere in this literature when survey/smart-card data isn't public: gravity-model demand proxy from population density (census/WorldPop) + POI density (OpenStreetMap) + published ridership figures where available. Xi'an and Chengdu likely have more accessible open transit/smart-city data.

**Next concrete step:** research actual open-data availability for all four cities (OSM extracts, national statistics offices, published feasibility studies e.g. JICA/ADB for Hanoi/HCMC, Chengdu smart-city open data portal) before finalizing the data-acquisition plan.

## 5. Experimental Design

- **Baselines** (all three MNEP reference papers, reimplemented): MetroGNN (single-objective GNN), Zhang et al. multi-objective actor-critic (no GNN), Michailidis et al. tabular RL.
- **Metrics**: OD demand satisfied (%), social equity (Gini/Theil index over station accessibility), radiation accessibility, training episodes to convergence, estimated CO2 (Michailidis et al.'s carbon-accounting method), wall-clock GPU time.
- **Ablations**: with/without cross-city transfer curriculum; with/without multi-objective (vs. demand-only reward); GNN state vs. flat/tabular state.

## 6. Manuscript Structure & Journal Fit

Maps to *Railway Engineering Science* scope items: "Design theory and construction technology," "Cutting-edge technologies," "Environmental impact and sustainability" (carbon-cost angle).

Structure: Introduction → Related Work (the 4 reference papers) → Problem Formulation (MNEP, multi-objective) → Method (GNN policy, Tchebycheff reward, transfer/efficiency layer) → Case Studies & Data (4 cities) → Experiments & Results → Discussion (scale-dependent guidance, limitations) → Conclusion.

## 7. Implementation Plan

- Environment: conda env `py313`, PyTorch with CUDA (NVIDIA GPU available) for GNN/actor-critic training.
- Repo layout (proposed):
  - `data/<city>/` — per-city graph, OD/demand proxy, POI/population inputs
  - `src/gnn/` — heterogeneous graph encoder
  - `src/rl/` — policy, Tchebycheff reward, transfer/curriculum, replay/shaping
  - `baselines/` — reimplementations of MetroGNN, Zhang et al., Michailidis et al.
  - `experiments/` — per-city training configs, ablation configs
  - `scripts/` — data acquisition/preprocessing per city

## 8. Risks / Open Questions

- OD demand data for Hanoi/HCMC not yet confirmed available (see §4) — needs research pass before committing to data pipeline.
- Reimplementing 3 baseline papers faithfully enough for fair comparison is nontrivial; may need to reach out to authors for code (MetroGNN and Michailidis et al. both mention public code/repos in-paper).
- Chengdu's scale (~500+ km, hundreds of stations) may push GNN training cost/time significantly higher than the other three cities — needs early feasibility check before full experimental run.
