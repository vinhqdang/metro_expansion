# Reproducing the Xi'an experimental results

`external/` is gitignored (third-party repos, cloned for inspection/reuse under their own licenses -- see `docs/data_sources.md`). This file records exactly what was done to get real numbers out of them, so the results in `manuscript/main.tex` Table 2 can be regenerated.

## Tabular RL baseline (Michailidis et al.)

```
git clone https://github.com/dimichai/tabular-tndp.git external/tabular-tndp
cd external/tabular-tndp
git submodule update --init --depth 1   # pulls envs/mo-tndp
cd envs/mo-tndp && pip install --no-deps -e .   # registers motndp_xian-v0 etc.
pip install wandb codecarbon mo-gymnasium
mkdir carbon_logs   # codecarbon.EmissionsTracker fails if this doesn't already exist
```

**Upstream bug patched** (`external/tabular-tndp/qlearning_tndp.py`, `QLearningTNDP.test()`): the method took `state` from the raw `env.reset()`/`env.step()` observation (a `MultiBinary` vector) instead of `info['location_grid_index']` (a scalar grid index) as their own `train()` method correctly does two methods above it -- `self.Q[state, :]` then either indexed nonsensically or raised `IndexError`. Confirmed via reading `train()`'s working code (which uses `info['location_grid_index']`) and a commented-out line in `test()` that reveals the same intent. Patched to match `train()`'s convention; the training loop itself was already correct and required no changes.

Reproduce:
```
cd external/tabular-tndp
WANDB_MODE=offline python train_qlearning.py --env xian --nr_stations 20 --train_episodes 500 --test_episodes 10 --seed <SEED>
```
for `SEED` in `{42, 1, 2, 3, 7}`. `WANDB_MODE=offline` avoids needing a wandb account -- the script otherwise calls `wandb.init()` unconditionally (its own `--no_log` flag is parsed but never actually wired to anything, a second, separate upstream bug not worth patching since offline mode is a sufficient workaround).

**Results** (`od_type=pct`, `reward_type=max_efficiency`, existing lines included): demand satisfied (%) per seed -- 42: 1.837, 1: 1.552, 2: 1.888, 3: 0.783, 7: 0.750. Mean 1.362 ± 0.558 (n=5, sample std).

**Independent re-verification (2026-08-25):** re-ran all 5 seeds from a fresh clone and re-derived the demand figure from scratch via a standalone script (`docs/xian_deterministic_regen_logs/`) that loads each seed's saved Q-table and rolls out its own greedy policy, using its own accumulation of the environment's per-group reward vector -- not by calling or trusting `QLearningTNDP.test()` at all. All 5 seeds reproduced exactly: 42: 1.8365, 1: 1.5520, 2: 1.8878, 3: 0.7830, 7: 0.7499 (mean 1.3618). This also surfaced a labeling nuance worth recording here in full since the manuscript only summarizes it: "Demand (%)" as reported by `calculate_reward('max_efficiency')` (`reward.sum() * 100`) is the *sum* of 5 already-normalized per-group percentages, not a single demand-mass-weighted city-wide percentage -- e.g. for seed 42's rollout, the true weighted-average percentage (using `city.group_od_sum` to un-normalize and re-aggregate) is 0.362%, versus the reported sum-of-fractions figure of 1.836% (roughly the number of groups apart). `scripts/train_xian.py` computes "Demand (%)" via the identical sum-of-group-fractions convention (`reward_vector.sum()`, inherited from the same environment), so the tabular-vs-GNN *comparison* in the manuscript's Table 1 is unaffected (both sides use the same definition) -- only the absolute number's literal interpretation needed this clarification.

## Genetic algorithm baseline (also Michailidis et al.'s code release)

`external/tabular-tndp/ga_tndp.py` / `run_ga.py` ship a genetic algorithm for the same
MO-TNDP problem (roulette-wheel selection, single-point crossover over candidate lines),
supporting `--env xian` directly. Not used anywhere in this project before 2026-08-25,
added to `manuscript/main.tex` Table 1 in response to the domain reviewer's observation
that every prior baseline was an RL method, leaving out the classical TNDP literature
(Section~2.1) that is the field's actual deployed alternative.

**Instrumentation added** (`ga_tndp.py`'s `run()`, not an upstream bug fix): the method
never returns or stores `best_episode_reward`/`best_episode_cells` outside its own local
scope, and only exposes them via `wandb.log`, which is unhelpful when running offline.
Added `self.best_episode_reward = ...` / `self.best_episode_cells = ...` assignments
inside the per-generation loop so a driver script can read the result after `.run()`
returns.

**Second upstream `--no_log` wiring bug found**, distinct from `qlearning_tndp.py`'s
already-documented one: `run()` calls `wandb.config['reward_type'] = reward_type`
unconditionally (not guarded by `if self.log:`, unlike its other wandb calls), so it
raises even with `log=False` unless `wandb.init()` has been called first. Worked around
in `run_ga_eval.py` (below) via `wandb.init(mode="disabled")` rather than patching the
library file, since `mode="disabled"` no-ops every wandb call without needing a real run.

Reproduce (`docs/xian_ga_baseline_logs/run_ga_eval.py`, copied from
`external/tabular-tndp/run_ga_eval.py`):
```
cd external/tabular-tndp
python run_ga_eval.py <SEED> 50   # population 100, 50 generations (hardcoded default in run_one())
```
for `SEED` in `{42, 1, 2, 3, 7}`.

**Results**: demand satisfied (%) per seed -- 42: 2.267, 1: 2.339, 2: 2.640, 3: 2.310,
7: 2.333. Mean 2.378 ± 0.149 (n=5, sample std). Wall-clock ~300s/seed on CPU (no GPU
needed, consistent with this method requiring none by design). This **beats both** the
tabular RL baseline (1.36%) and all three of our GNN-based variants (0.16--0.61%) --
reported in full in `docs/xian_ga_baseline_logs/results.txt`. Budget (population 100,
50 generations = 10,100 environment rollouts total) was chosen for this revision's time
constraints, not tuned for maximum GA performance; demand was still improving at the
budget's end, so a larger budget would plausibly widen this gap further, not close it.

## Our method (full + 2 ablations)

No external patching needed -- `scripts/train_xian.py` reuses the same cloned `external/tabular-tndp/envs/mo-tndp` environment mechanics (action masking, episode structure) but replaces their tabular Q-learning with our own GNN encoder + multi-objective actor-critic (`src/gnn`, `src/rl`).

```
python scripts/train_xian.py --train_episodes 500 --nr_stations 20 --seed <SEED> --ablation {full,single_objective,flat_encoder}
```

`full` is our complete method. `single_objective` and `flat_encoder` are principled ablations standing in for MetroGNN (su2024) and Zhang et al. (zhang2024) respectively, since neither has public code to run directly -- see the module docstring in `scripts/train_xian.py` for exactly what each ablation changes and why it approximates (not reproduces) the corresponding paper.

Each run prints results for 4 weight-vector settings (uniform, demand-only, equity-only, coverage-only) from the same trained model, plus an explicit sanity check on whether the policy's behavior actually differs across them -- included specifically because an early version of the training loop had a bug where it did not (see git history, "feat: implement and train the full method on Xi'an" commit message, for the two real bugs found and fixed this way).

**Deterministic regeneration (2026-08-25):** the current `scripts/train_xian.py` already sets `torch.use_deterministic_algorithms(True)` (with `CUBLAS_WORKSPACE_CONFIG` set first) and seeds the Gymnasium environment once at construction. Re-ran the full 3-method × 5-seed sweep (15 runs) under this configuration on CPU (this machine has no NVIDIA GPU; the manuscript's earlier tables were collected on an RTX A2000), confirmed byte-identical outputs across 3 independent re-runs of `full`/seed 42, and used the resulting uniform-weight greedy-eval numbers to regenerate `manuscript/main.tex`'s Table 1 (`tab:results`). Total wall-clock for all 15 runs: well under 30 minutes (each run 20-70s). Raw per-run logs and the `compute_stats.py` script used to derive the reported means/sample-stds are in `docs/xian_deterministic_regen_logs/`. Note this regeneration changed compute backend (GPU → CPU) at the same time as applying the determinism fix, so the change from the previous table's numbers cannot be cleanly attributed to the fix alone -- the manuscript states this explicitly. `tab:weight-sensitivity`, `tab:training-curve`, and `fig:xian-map` were **not** regenerated in this pass and still describe the earlier, non-deterministic-GPU seed-42 checkpoint.
