# metro_expansion

Research code and manuscript for a submission to [*Transportmetrica B: Transport Dynamics*](https://www.tandfonline.com/journals/ttrb20) (Taylor & Francis): a GNN-based, multi-objective reinforcement learning method for urban rail transit network expansion, evaluated across multiple Asian cities (Hanoi, Ho Chi Minh City, Xi'an, Chengdu, extensible to more).

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
   applied to transportation network design), requiring no reframing. Submitted, and
   desk-rejected there too, with no reason given -- the same pattern later repeated at
   Part B (step 6). (Note: `manuscript/main.tex`'s header comment previously stated this
   attempt "did not proceed"; that comment predates confirmation of the actual rejection
   and is stale -- corrected here.)
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
   domain/perspective branch's Major Revision path. Both Elsevier Transportation Research
   venues tried so far (Part C, Part B) have now desk-rejected the paper without a
   substantive reason. Decision (2026-08-31): stop targeting the Transportation Research
   family; retarget to a Q1/Q2 journal (SJR/JCR) in the transportation/AI-for-transportation
   space with no mandatory submission fee and no mandatory APC (subscription or hybrid
   model, not gold/mandatory-OA) for the next attempt. Candidate list to be researched and
   recorded before the next retarget.
7. Retargeted to *Transportmetrica B: Transport Dynamics* (Taylor & Francis) (2026-08-31),
   selected from a verified shortlist (Q2, JCR 2025 IF 4.2; hybrid Open Select publishing,
   no mandatory submission fee or APC) over Transportmetrica A, Transportation Science, and
   IEEE T-ITS. The strongest signal for this specific choice: this journal already
   published Holliday, El-Geneidy, and Dudek (2025), "Learning heuristics for transit
   network design and improvement with deep reinforcement learning" -- a GNN+RL
   transit-network-design paper closely matching this manuscript's own method and domain --
   which is closer to a demonstrated-fit precedent than either Transportation Research venue
   had going in. Unlike Part B, Transportmetrica B runs single-anonymized review (reviewer
   anonymous to author; author identity visible to the reviewer), so the double-blind
   anonymization apparatus from step 4 was reverted rather than carried forward. Converted
   `manuscript/main.tex` from Elsevier's `elsarticle` to Taylor & Francis's `interact`
   class with the `tfcad.bst` Chicago author-date bibliography style (the
   `interactcadlatex` bundle, T&F's official LaTeX template for this journal), restored
   the real author/CRediT/data-availability content in place of the anonymized versions,
   added the newly-required Funding and Declaration of generative AI use sections, moved
   References ahead of the Appendices to match the journal's stated section order, and
   condensed the abstract from 409 words to ~160 to approach the journal's 150-word
   guideline (still not exact-compliant; a further tightening pass is open). Removed the
   now-unused Elsevier (`elsarticle.cls`/`elsarticle-harv.bst`), double-blind
   (`titlepage.tex`/`.pdf`), and long-stale Springer Nature (`bst/`, `template/`) files.
   Compiles cleanly (`pdflatex` x3 + `bibtex`) to a 50-page PDF with no undefined
   references. **Open item, not resolved by this retarget:** the main text (Introduction
   through Conclusion, excluding tables/figures/references) runs to roughly 14,200 words
   against the journal's stated typical limit of 10,000 -- a genuine editorial-cut task,
   flagged in the cover letter and in `main.tex`'s header comment, not attempted here.

## Layout

- `manuscript/` -- LaTeX manuscript (`main.tex`) and `cover_letter.tex`, on Taylor &
  Francis's `interact` document class (Interact-CAD bundle: `interact.cls`, `tfcad.bst`
  Chicago author-date bibliography style, both the genuine T&F-distributed files, not a
  shim) per Transportmetrica B's LaTeX Instructions for Authors. Declarations sections
  (CRediT authorship, Disclosure statement, Funding, Declaration of generative AI use,
  Data availability) follow T&F's required headings; References are placed before the
  Appendices per the journal's stated structure. Single-anonymized submission (see step 7
  above): no author-identity suppression and no separate title-page file, unlike the
  double-blind Elsevier attempt this replaced. Build with
  `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` from within
  `manuscript/` (`cover_letter.tex` needs only a single `pdflatex` pass, no bibliography).
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
