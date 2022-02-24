import argparse
import os
import pickle

import numpy as np
import optproblems.wfg
import matplotlib.pyplot as plt
import pyDOE2
from multiprocessing import Pool
import optuna


class WFG:
    def __init__(self, num_wfg: int) -> None:
        self.function = {
            1: optproblems.wfg.WFG1,
            2: optproblems.wfg.WFG2,
            3: optproblems.wfg.WFG3,
            4: optproblems.wfg.WFG4,
            5: optproblems.wfg.WFG5,
            6: optproblems.wfg.WFG6,
            7: optproblems.wfg.WFG7,
            8: optproblems.wfg.WFG8,
            9: optproblems.wfg.WFG9
        }[num_wfg](num_objectives, num_variables, k)

    def __call__(self, trial):
        x = []
        for i in range(1, num_variables + 1):
            x.append(trial.suggest_uniform(f"x{i}", 0.0, 2.0 * i))

        fx = np.array(self.function.objective_function(x))
        return fx


def solve(seed):
    sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler(
        n_startup_trials=n_startup_trials,
        n_ehvi_candidates=24,
        consider_endpoints=True,
        seed=seed
    )
    study = optuna.multi_objective.create_study(
        sampler=sampler, directions=["minimize"] * num_objectives
    )

    initial_samples = pyDOE2.lhs(
        num_variables, samples=n_startup_trials, criterion='maximin',
        random_state=np.random.RandomState(seed)
    )
    denormalize = lambda i, v: 2.0 * i * v
    for s in initial_samples:
        study.enqueue_trial({f'x{i}': denormalize(i, v) for i, v in enumerate(s, start=1)})

    study.optimize(WFG(num_wfg), n_trials=250, n_jobs=1)
    trials = [{
        'values': trial.values,
        'datetime_start': trial.datetime_start,
        'datetime_complete': trial.datetime_complete
    } for trial in study.get_trials()]
    return trials

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_wfg', default=4, type=int)
    parser.add_argument('--num_objectives', default=2, type=int)
    parser.add_argument('--num_variables', default=3, type=int)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    num_wfg = args.num_wfg
    num_objectives = args.num_objectives
    num_variables = args.num_variables
    k = num_objectives - 1
    n_startup_trials = 11 * num_variables - 1
    reference_point = [3, 5, 7, 9][:num_objectives]
    seed = args.seed

    results = solve(seed)

