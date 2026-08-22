# metro_expansion

Research code and manuscript for a submission to [*Transportation Research Part C: Emerging Technologies*](https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies) (Elsevier): a GNN-based, multi-objective reinforcement learning method for urban rail transit network expansion, evaluated across multiple Asian cities (Hanoi, Ho Chi Minh City, Xi'an, Chengdu, extensible to more).

**Submission history:**
1. Originally targeted at *Railway Engineering Science* (Springer / Southwest Jiaotong
   University); desk-rejected there (2026-08-22), most likely for scope fit.
2. Retargeted to *Computational Economics* (Springer), reframing the problem as a
   public-investment resource-allocation question. A five-reviewer simulated review panel
   (`academic-paper-reviewer` skill; EIC + 3 peer reviewers + Devil's Advocate) converged
   independently on Reject/Desk Reject for that venue: the economics framing existed only
   in the cover letter and one undeveloped sentence of the manuscript, with no welfare
   function, monetization, discounting, or economics-literature engagement anywhere in the
   actual paper -- a vocabulary relabeling over unchanged RL/GNN content, not a genuine
   disciplinary reframing. Full review reports and the editorial synthesis are recorded in
   this project's session history (2026-08-22); not persisted as a separate file.
3. Retargeted again, this time to *Transportation Research Part C* -- the venue the
   review panel identified as the actual natural fit for the paper's real content
   (AI/ML/RL methods applied to transportation network design), requiring no reframing.

## Layout

- `manuscript/` -- LaTeX manuscript and `cover_letter.tex`, on Elsevier's `elsarticle`
  class (author-year mode: `\documentclass[authoryear,preprint,12pt]{elsarticle}`,
  `elsarticle-harv.bst`). Author-year rather than TRC's house numbered style
  (`elsarticle-num`) because the manuscript's prose relies heavily on named in-text
  citations (`\citeauthor{su2024}`-style), which `elsarticle-num.bst` does not resolve
  under natbib; swap to `[number]` + `elsarticle-num.bst` at production stage if the
  journal requires it. `elsarticle.cls`/`elsarticle-harv.bst` here are the genuine,
  freely-redistributable Elsevier files (via TeX Live), not a shim. Declarations sections
  (CRediT authorship, competing interest, data availability, acknowledgements) follow
  Elsevier's standard headings rather than Springer's itemized "Declarations" block.
  **Still open before actual submission:** confirm TRC's single- vs double-blind policy
  (`elsarticle.cls` has a `doubleblind` option that suppresses author info from the
  rendered PDF if needed) and whether a Highlights file (3-5 bullet points) is required at
  the submission portal. Build with
  `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
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
