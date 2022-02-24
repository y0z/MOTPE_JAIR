import subprocess
import numpy as np
import os

import optuna


def objective(trial):
    num_blocks = trial.suggest_int("num_blocks", 1, 3)
    num_filters_1 = trial.suggest_int("num_filters_1", 16, 256)
    batch_norm_1 = trial.suggest_categorical("batch_norm_1", [0, 1])
    if num_blocks >= 2:
        num_filters_2 = trial.suggest_int("num_filters_2", 16, 256)
        batch_norm_2 = trial.suggest_categorical("batch_norm_2", [0, 1])
    if num_blocks >= 3:
        num_filters_3 = trial.suggest_int("num_filters_3", 16, 256)
        batch_norm_3 = trial.suggest_categorical("batch_norm_3", [0, 1])
    pooling = trial.suggest_categorical("pooling", ["Max", "Average"])
    dropout_rate = trial.suggest_uniform("dropout_rate", 0, 0.9)
    num_units = trial.suggest_int("num_units", 16, 4096, log=True)
    learning_rate = trial.suggest_loguniform("learning_rate", 0.00001, 0.1)
    momentum = trial.suggest_uniform("momentum", 0.8, 1.0)

    dirname = os.path.dirname(__file__)
    x = [
        "python3",
        dirname + "/cnn.py",
        "--num_blocks",
        str(num_blocks),
        "--num_filters_1",
        str(num_filters_1),
        "--batch_norm_1",
        str(batch_norm_1),
        "--pooling",
        pooling,
        "--dropout_rate",
        str(dropout_rate),
        "--num_units",
        str(num_units),
        "--learning_rate",
        str(learning_rate),
        "--momentum",
        str(momentum)
    ]
    if num_blocks >= 2:
        x.append("--num_filters_2")
        x.append(str(num_filters_2))
        x.append("--batch_norm_2")
        x.append(str(batch_norm_2))
    if num_blocks >= 3:
        x.append("--num_filters_3")
        x.append(str(num_filters_3))
        x.append("--batch_norm_3")
        x.append(str(batch_norm_3))

    # Run cnn.py
    proc = subprocess.run(x, stdout=subprocess.PIPE)
    outputs = proc.stdout.decode("utf8").split("\n")[-2].split(" ")
    f1 = float(outputs[0])
    f2 = float(outputs[1])
    return f1, f2


if __name__ == "__main__":
    sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler(
        n_ehvi_candidates=24,
        consider_endpoints=True,
    )
    study = optuna.multi_objective.create_study(
        sampler=sampler, directions=["minimize", "minimize"]
    )

    n_startup_trials = 50
    for _ in range(n_startup_trials):
        num_blocks = np.random.randint(1, 4)
        num_filters_1 = int(np.random.randint(16, 257))
        batch_norm_1 = np.random.randint(0, 2)
        num_filters_2 = int(np.random.randint(16, 257))
        batch_norm_2 = np.random.randint(0, 2)
        num_filters_3 = int(np.random.randint(16, 257))
        batch_norm_3 = np.random.randint(0, 2)
        pooling = ["Max", "Average"][np.random.randint(2)]
        dropout_rate = np.random.rand() * 0.9
        num_units = int(round(np.exp2(4 + 8 * np.random.rand())))
        learning_rate = 10 ** (-5 + 4 * np.random.rand())
        momentum = np.clip(0.8 + 0.2 * np.random.rand(), 0, 1)
        x = {}
        x["num_blocks"] = num_blocks
        x["num_filters_1"] = num_filters_1
        x["batch_norm_1"] = batch_norm_1
        x["num_filters_2"] = num_filters_2
        x["batch_norm_2"] = batch_norm_2
        x["num_filters_3"] = num_filters_3
        x["batch_norm_3"] = batch_norm_3
        x["pooling"] = pooling
        x["dropout_rate"] = dropout_rate
        x["num_units"] = num_units
        x["learning_rate"] = learning_rate
        x["momentum"] = momentum

        x_ = x.copy()
        if num_blocks < 2:
            del x_["num_filters_2"]
            del x_["batch_norm_2"]
        if num_blocks < 3:
            del x_["num_filters_3"]
            del x_["batch_norm_3"]

        study.enqueue_trial(x_)

    study.optimize(objective, n_trials=150)
    fig = optuna.multi_objective.visualization.plot_pareto_front(study)
    fig.show()
