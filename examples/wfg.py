import argparse
import numpy as np
import optproblems.wfg
import optuna
import pyDOE2
import time


class WFG:
    def __init__(self, n_wfg: int, seed: int) -> None:
        self.function = {
            1: optproblems.wfg.WFG1,
            2: optproblems.wfg.WFG2,
            3: optproblems.wfg.WFG3,
            4: optproblems.wfg.WFG4,
            5: optproblems.wfg.WFG5,
            6: optproblems.wfg.WFG6,
            7: optproblems.wfg.WFG7,
            8: optproblems.wfg.WFG8,
            9: optproblems.wfg.WFG9,
        }[n_wfg](n_objectives, n_variables, k)
        self.rng = np.random.RandomState(seed)

    def __call__(self, trial):
        x = []
        for i in range(1, n_variables + 1):
            x.append(trial.suggest_uniform(f"x{i}", 0.0, 2.0 * i))

        fx = np.array(self.function.objective_function(x))

        # Random wait to simulate expensive objectives
        #wait_time = max(0, self.rng.normal(60, 15))
        #time.sleep(wait_time)

        return fx


class GammaFunction:
    def __init__(self, gamma):
        self._gamma = gamma

    def __call__(self, x: int, _: int) -> int:
        return int(np.floor(self._gamma * x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_wfg", default=4, type=int)
    parser.add_argument("--n_objectives", default=2, type=int)
    parser.add_argument("--n_variables", default=3, type=int)
    parser.add_argument("--n_startup_trials", default=32, type=int)
    parser.add_argument("--gamma", default=0.10, type=float)
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--n_jobs", default=1, type=int)
    parser.add_argument("--n_trials", default=250, type=int)
    args = parser.parse_args()

    # Configuration
    n_wfg = args.n_wfg
    n_objectives = args.n_objectives
    n_variables = args.n_variables
    k = args.k
    n_startup_trials = args.n_startup_trials
    gamma = args.gamma
    n_jobs = args.n_jobs
    n_trials = args.n_trials
    seed = 42

    # Prepare sampler and study
    # MOTPE
    sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler(
        n_ehvi_candidates=24, consider_endpoints=True, gamma=GammaFunction(gamma),
        seed=seed
    )
    # NSGA-II
    '''
    sampler = optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler(
        population_size=100,
        seed=seed
    )
    '''

    study = optuna.multi_objective.create_study(
        sampler=sampler, directions=["minimize"] * n_objectives
    )

    # Initialization with Latin Hypercube Sampling
    initial_samples = pyDOE2.lhs(
        n_variables, samples=n_startup_trials, criterion="maximin",
        random_state=np.random.RandomState(seed + 1234)
    )
    denormalize = lambda i, v: 2.0 * i * v
    for s in initial_samples:
        study.enqueue_trial(
            {f"x{i}": denormalize(i, v) for i, v in enumerate(s, start=1)}
        )

    # Optimize
    study.optimize(
        WFG(n_wfg, seed + 5678),
        n_trials=n_trials,
        n_jobs=n_jobs,  # Number of workers for parallelization
    )

    # Print results
    '''
    for t in study.trials:
        print(t.values)
        print(t.params)
    '''

    # Plot results
    fig = optuna.multi_objective.visualization.plot_pareto_front(study)
    fig.show()
