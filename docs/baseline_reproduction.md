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

## Our method (full + 2 ablations)

No external patching needed -- `scripts/train_xian.py` reuses the same cloned `external/tabular-tndp/envs/mo-tndp` environment mechanics (action masking, episode structure) but replaces their tabular Q-learning with our own GNN encoder + multi-objective actor-critic (`src/gnn`, `src/rl`).

```
python scripts/train_xian.py --train_episodes 500 --nr_stations 20 --seed <SEED> --ablation {full,single_objective,flat_encoder}
```

`full` is our complete method. `single_objective` and `flat_encoder` are principled ablations standing in for MetroGNN (su2024) and Zhang et al. (zhang2024) respectively, since neither has public code to run directly -- see the module docstring in `scripts/train_xian.py` for exactly what each ablation changes and why it approximates (not reproduces) the corresponding paper.

Each run prints results for 4 weight-vector settings (uniform, demand-only, equity-only, coverage-only) from the same trained model, plus an explicit sanity check on whether the policy's behavior actually differs across them -- included specifically because an early version of the training loop had a bug where it did not (see git history, "feat: implement and train the full method on Xi'an" commit message, for the two real bugs found and fixed this way).
