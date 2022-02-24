# README

This repository is the code base for the paper "Multiobjective Tree-structured Parzen Estimator".
The latest version of MOTPE is available in the latest version of [Optuna](https://optuna.org/).

## Files

- `optuna`
    - MOTPE implementation is `optuna/optuna/multi_objective/samplers/_motpe.py`
- `wfg.py`
    - Example code to run MOTPE on the WFG benchmark problems
- `cnn_design.py`
    - Example code to run MOTPE on the CNN design problem
- `cnn.py`
    - CNN implementation of the CNN design problem


## Setup Optuna

Run the following command to install the dependncies.
```sh
python3 setup.py install
```

## Run the WFG benchmark example

First, add `optuna` to `$PYTHONPATH`.
Then, run `wfg.py`.

```sh
PYTHONPATH=optuna:$PYTHONPATH python3 examples/wfg.py --n_wfg 4 --n_objectives 2 --n_variables 9 --k 1 --n_trials 250 --n_startup_trials 98 --gamma 0.1 --n_jobs 1
```

The command line options are as follows.
- `n_wfg`: wfg benchmark number (1--9)
- `n_objectives`: number of objectives
- `n_variables`: number of variables
- `k`: parameter k for the wfg problem
- `n_trials`: number of trials (evaluations)
- `n_startup_trials`: number of initial evaluations with Latin hypercube sampling
- `gamma`: gamma parameter for MOTPE
- `n_jobs`: number of workers for parallelization of MOTPE

The documentation of Optuna is helpful to understand the code.
[Optuna: A hyperparameter optimization framework Optuna 2.0.0 documentation](https://optuna.readthedocs.io/en/v2.0.0/)

## Run the CNN design example

Please install Tensorflow/Keras (version 2.2.0) following the official documentation ([TensorFlow](https://www.tensorflow.org/)).

Run `cnn_design.py`.

```sh
PYTHONPATH=optuna:$PYTHONPATH python3 examples/cnn_design.py
```
