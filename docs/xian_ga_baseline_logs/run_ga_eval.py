"""Driver for evaluating GATNDP (the classical genetic-algorithm baseline
already present in this cloned repo, envs/ga_tndp.py) on Xi'an, without
needing wandb -- used as the classical TNDP comparison point for
manuscript/main.tex's Table 1, per the domain reviewer's request for a
non-RL baseline.
"""
import sys
import time
from pathlib import Path

import envs  # noqa: F401
import mo_gymnasium as mo_gym
import numpy as np
import wandb
from ga_tndp import GATNDP
from motndp.city import City
from motndp.constraints import MetroConstraints

# ga_tndp.py's run() calls wandb.config[...] = ... unconditionally (not
# guarded by `if self.log`, unlike its other wandb calls) -- a second,
# separate upstream --no_log wiring bug beyond the one already documented
# for qlearning_tndp.py in docs/baseline_reproduction.md. Disabled mode
# no-ops every wandb call without needing a real run or network access.
wandb.init(mode="disabled")


def run_one(seed, generations, init_pop_size=100, nr_stations=20, nr_groups=5):
    city = City(Path("./envs/mo-tndp/cities/xian"), groups_file=f"price_groups_{nr_groups}.txt",
                ignore_existing_lines=False)
    env = mo_gym.make("motndp_xian-v0", city=city, constraints=MetroConstraints(city),
                       nr_stations=nr_stations, od_type="pct", chained_reward=False)
    algo = GATNDP(env, init_pop_size=init_pop_size, mutation_rate=0.9, crossover_rate=0.9,
                  generations=generations, nr_stations=nr_stations, nr_groups=nr_groups,
                  seed=seed, wandb_project_name="unused", wandb_experiment_name="unused",
                  log=False)
    t0 = time.time()
    algo.run(reward_type="max_efficiency")
    wall = time.time() - t0
    demand_pct = algo.best_episode_reward * 100  # sum-of-group-fractions convention, same as Table 1
    print(f"seed={seed} generations={generations} pop={init_pop_size} "
          f"best_demand_pct={demand_pct:.5f} wall_s={wall:.1f} "
          f"line_len={len(algo.best_episode_cells)}")
    return demand_pct, wall


if __name__ == "__main__":
    seed = int(sys.argv[1])
    generations = int(sys.argv[2])
    run_one(seed, generations)
