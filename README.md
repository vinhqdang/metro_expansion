# metro_expansion

Research code and manuscript for a submission to [*Transportation Research Part B: Methodological*](https://www.sciencedirect.com/journal/transportation-research-part-b-methodological) (Elsevier): a GNN-based, multi-objective reinforcement learning method for urban rail transit network expansion, evaluated across multiple Asian cities (Hanoi, Ho Chi Minh City, Xi'an, Chengdu, extensible to more).

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
3. Retargeted again, to *Transportation Research Part C* -- the venue the review panel
   identified as the actual natural fit for the paper's real content (AI/ML/RL methods
   applied to transportation network design), requiring no reframing. Not proceeded with.
4. Retargeted again, to *Transportation Research Part B* (same Elsevier `elsarticle`
   template as the Part C attempt, no format change needed) and converted to
   double-anonymized (double-blind) submission format (2026-08-25): `main.tex` now uses
   elsarticle's `[doubleblind]` class option plus manual anonymization of the CRediT and
   Data availability sections (the project's own GitHub URL identifies the author and is
   withheld from the blinded manuscript); a separate `manuscript/titlepage.tex`/`.pdf`
   carries the real author, affiliation, and repository link for editor/reviewer use.
5. A five-reviewer simulated review panel against the Part B target (2026-08-25; EIC + 3
   peer reviewers + Devil's Advocate) split: EIC and Devil's Advocate recommended Reject on
   venue-fit/evidentiary-completeness grounds (the paper's one head-to-head result loses to
   a simpler baseline; only 2 of 4 planned cities done); the methodology, domain, and
   perspective reviewers recommended Major Revision with concrete, actionable fixes.
   Editorial synthesis: Major Revision (last chance), gated on either bringing in the
   Chengdu result or holding the Part B attempt. Not persisted as a separate file (session
   history). Acted on immediately: moved the "exploratory tool, not decision-support
   system" framing into the abstract/intro; flagged the price-band equity metric's
   circularity at its definition; added exact sign-test stats to the curriculum claim;
   independently re-verified the tabular RL baseline from scratch (reproduces exactly);
   regenerated Table 1 under the deterministic-training fix that had been implemented but
   never applied (new numbers disagree with both predecessors on which architecture leads
   -- reported as the finding itself, not smoothed over); added a classical genetic-algorithm
   baseline (already in Michailidis et al.'s own code release, previously unused) that
   **beats every RL method in the paper, including ours**; and ablated the bus-signal claim
   (substantiated, narrower than originally stated -- demand only, 5/5 seeds). Raw logs for
   all new experiments are under `docs/xian_deterministic_regen_logs/`,
   `docs/xian_ga_baseline_logs/`, and `docs/hcmc_bus_ablation_logs/`. Chengdu/Hanoi and the
   remaining tables' deterministic regeneration are still open (Future Work).
6. Actual Part B editorial decision received (2026-08-31): rejected at the journal's
   initial-review stage ("could be potentially interesting to a number of journals but...
   not deemed to be the best fit with Part B"), with no reviewer-specific comments attached
   -- a suitability/venue-fit desk reject, not a peer-review outcome. This confirms the
   EIC/Devil's Advocate branch of the step-5 simulated panel rather than the methodology/
   domain/perspective branch's Major Revision path. The step-3 candidate (*Transportation
   Research Part C*) remains the panel-identified natural fit requiring no reframing; not
   yet acted on.

## Layout

- `manuscript/` -- LaTeX manuscript (`main.tex`), separate anonymized `titlepage.tex`, and
  `cover_letter.tex`, on Elsevier's `elsarticle` class (author-year mode:
  `\documentclass[doubleblind,authoryear,preprint,12pt]{elsarticle}`, `elsarticle-harv.bst`).
  Author-year rather than Part B's more common house numbered style (`elsarticle-num`)
  because the manuscript's prose relies heavily on named in-text citations
  (`\citeauthor{su2024}`-style), which `elsarticle-num.bst` does not resolve under natbib;
  swap to `[number]` + `elsarticle-num.bst` at production stage if the journal requires it.
  `elsarticle.cls`/`elsarticle-harv.bst` here are the genuine, freely-redistributable
  Elsevier files (via TeX Live), not a shim. Declarations sections (CRediT authorship,
  competing interest, data availability, acknowledgements) follow Elsevier's standard
  headings rather than Springer's itemized "Declarations" block.
  **Double-anonymized submission format:** Part B runs double-blind review. `main.tex`'s
  `[doubleblind]` class option suppresses `\author`/`\ead`/`\affiliation` from the rendered
  PDF automatically, but does not touch free-text sections -- the CRediT statement (author
  name withheld) and Data availability statement (project GitHub URL withheld, since it
  encodes the author's name) are manually anonymized in the source. `titlepage.tex` is the
  separate, unblinded file carrying the real author/affiliation/CRediT/data-link
  information, uploaded alongside the anonymized manuscript per Elsevier's submission
  requirements. **Still open before actual submission:** whether a Highlights file (3-5
  bullet points) is required at the submission portal. Build each document independently
  with `pdflatex <file>.tex && bibtex <file> && pdflatex <file>.tex && pdflatex <file>.tex`
  (bibtex step only needed for `main.tex`) from within `manuscript/`.
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
