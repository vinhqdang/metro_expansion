# metro_expansion

Research code and manuscript for a submission to [*Computational Economics*](https://link.springer.com/journal/10614/aims-and-scope) (Springer, official journal of the Society for Computational Economics): a GNN-based, multi-objective reinforcement learning method for urban rail transit network expansion, evaluated across multiple Asian cities (Hanoi, Ho Chi Minh City, Xi'an, Chengdu, extensible to more).

The paper frames metro network expansion as a public-investment resource-allocation
problem -- trading aggregate transportation-demand efficiency against equitable access
across neighborhoods under a fixed budget -- solved with agent-based/ML computational
methods (graph representation learning, multi-objective reinforcement learning,
Tchebycheff scalarization), which is why it targets *Computational Economics* rather
than a transportation-engineering venue.

**Submission history:** originally targeted at *Railway Engineering Science* (Springer /
Southwest Jiaotong University); desk-rejected there (2026-08-22), most likely for scope
fit (that journal's remit is rail engineering design/operations, not the resource-allocation/
welfare framing this paper leads with). Retargeted to *Computational Economics*
accordingly; see `manuscript/cover_letter.tex`.

## Layout

- `manuscript/` -- LaTeX manuscript and `cover_letter.tex`. Reformatted from the earlier
  Springer Nature `sn-jnl`/Vancouver-numbered template (*Railway Engineering Science*'s
  house style) to `svjour3` with the `spbasic` (author-year) bibliography style, matching
  *Computational Economics*. Three passages that referenced the old target journal's
  publisher (Southwest Jiaotong University/Chengdu) and scope wording were also corrected.
  **Before actual submission:** `manuscript/svjour3.cls` here is a local preview shim (the
  real Springer `svjour3.cls` isn't distributable via CTAN and wasn't reachable from this
  environment) -- it approximates single-column Springer layout well enough to proofread
  content, but must be replaced with the genuine class from the journal's own LaTeX author
  template before submission; `main.tex`'s source targets the real class and shouldn't need
  changes when it's dropped in. `spbasic.bst` is the genuine, freely-redistributable style.
  Build with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
  from within `manuscript/`.
- `docs/superpowers/specs/` -- design spec for the project (research gap, contribution, method, case studies, experimental design).
- `src/gnn/` -- heterogeneous graph state encoder (spatial-contiguity + OD-flow relations).
- `src/rl/` -- multi-objective actor-critic policy, Tchebycheff reward scalarization, prioritized replay buffer.
- `src/data/` -- per-city data loading/preprocessing (network topology, demand proxies, candidate regions).
- `baselines/` -- reimplementations of the three reference RL methods for MNEP (MetroGNN, Zhang et al. 2024, Michailidis et al. 2026).
- `data/<city>/` -- per-city raw/processed data (not committed where licenses restrict redistribution -- see `.gitignore`).
- `experiments/` -- per-city training configs and ablation configs.
- `tests/` -- unit tests (`pytest tests/`).

## Environment

Conda env `py313` (Python 3.13, CUDA-enabled PyTorch). See `environment.yml` / `requirements.txt`. GPU: NVIDIA RTX A2000 8GB (or any CUDA-capable GPU).

```
pytest tests/
```

## Scope note

The method's action space and object of design are rail transit only (metro, tram/light-rail). Where a city's existing rail network is too sparse to define a well-posed expansion problem (e.g. Hanoi, Ho Chi Minh City), the existing bus network is used only as an auxiliary signal (candidate-region definition, demand proxy) -- never as something the policy designs or evaluates. See `docs/superpowers/specs/2026-08-18-metro-expansion-manuscript-design.md` §9 for the rationale.
