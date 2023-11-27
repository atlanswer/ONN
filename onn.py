# %% Import
from typing import cast
import numpy as np
import numpy.typing as npt
from opytimark.markers.n_dimensional import Griewank  # pyright: ignore[reportMissingTypeStubs]
import torch
import torch.nn as nn
import torch.optim as optim
import time
from loguru import logger
import os

# import csv
from antcal.application.helper import new_hfss_session  # pyright: ignore [reportMissingTypeStubs]

cwd = os.getcwd()
save_path = os.path.join(cwd, "save")
logger.add("log.log")


rng = np.random.default_rng()

# %% Init
# Optimization parameters
n_dims = 2
s_init = 4
n_candidates = 4
n_new_candidates = 1
n_predictors = 5
n_iters = 200
alpha_mut_ratio = 0.4
n_dims_mut = int(alpha_mut_ratio * n_dims) + 1
n_opts = 10

# Hyperparameters

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MODEL_DIMS = 100 * n_dims
EPOCHS = 2 * int(s_init / BATCH_SIZE) + 1

# Cost function
cost_fn = Griewank(dims=n_dims)
var_bounds = [(-600.0, 600.0)] * n_dims

h1 = new_hfss_session()

def gen_init_X(
    n_dims: int, s_init: int, var_bounds: list[tuple[float, float]]
) -> npt.NDArray[np.float64]:
    lower_bounds = np.array([i[0] for i in var_bounds])
    upper_bounds = np.array([i[1] for i in var_bounds])
    v_r = rng.random((s_init, n_dims))
    return lower_bounds + v_r * (upper_bounds - lower_bounds)


def v_cost_fn(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.apply_along_axis(cost_fn, 1, X)


def gen_init_predictors(
    n_predictors: int, dims_in: int
) -> tuple[list[nn.Sequential], list[optim.Adam]]:
    predictors = []
    optimizers = []
    for _ in range(n_predictors):
        net = nn.Sequential(
            nn.Linear(dims_in, MODEL_DIMS),
            nn.LeakyReLU(),
            nn.Linear(MODEL_DIMS, MODEL_DIMS),
            nn.LeakyReLU(),
            nn.Linear(MODEL_DIMS, 1),
        )
        predictors.append(net)
        optimizers.append(optim.Adam(net.parameters(), lr=LEARNING_RATE))
    return predictors, optimizers  # pyright: ignore[reportUnknownVariableType]


def update_predictors(
    predictors: list[nn.Sequential],
    optimizers: list[optim.Adam],
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
):
    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(Y)
    loss_fn = nn.MSELoss()
    errors = []
    for i in range(len(predictors)):
        net = predictors[i]
        optimizer = optimizers[i]
        prediction = net(x_tensor.float())
        loss = loss_fn(prediction, y_tensor.float().view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        errors.append(loss.detach())
    avg_error = torch.mean(torch.stack(errors))
    return predictors, optimizers, avg_error


def mutate(
    candidate: npt.NDArray[np.float64],
    global_min: np.float64,
    global_max: np.float64,
    n_dims_mut: int,
):
    idx = rng.choice(np.arange(len(candidate)), n_dims_mut)
    candidate_mut = candidate
    for i in range(n_dims_mut):
        candidate_mut[idx[i]] = global_min + rng.random() * (global_max - global_min)
    return candidate_mut


def gen_candidates(
    X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], n_candidates: int
):
    idx = np.argpartition(Y, n_candidates - 1)
    X_candidates = []
    Y_candidates = []
    for i in range(n_candidates):
        X_candidates.append(X[idx[i]])
        Y_candidates.append(Y[idx[i]])
    X_candidates = cast(npt.NDArray[np.float64], np.array(X_candidates))
    Y_candidates = cast(npt.NDArray[np.float64], np.array(Y_candidates))
    X_min = X_candidates.min()
    X_max = X_candidates.max()
    X_candidates_mutated = []
    for candidate in X_candidates:
        X_candidates_mutated.append(mutate(candidate, X_min, X_max, n_dims_mut))
    X_candidates_mutated = cast(npt.NDArray[np.float64], np.array(X_candidates_mutated))
    return X_candidates_mutated


def find_new_candidates(
    X_candidates_mut: npt.NDArray[np.float64],
    predictors: list[nn.Sequential],
    n_predictors: int,
    n_new_candidates: int,
):
    new_candidates = []
    predictor = predictors[rng.integers(0, n_predictors - 1)]
    predictor.eval()
    with torch.no_grad():
        prediction = predictor(torch.from_numpy(np.array(X_candidates_mut)).float())
        prediction = cast(npt.NDArray[np.float64], prediction.numpy())
    min_idx = (
        np.argpartition(prediction.transpose(), n_new_candidates - 1)
    ).transpose()
    for i in range(n_new_candidates):
        new_candidates.extend(X_candidates_mut[min_idx[i]])
    new_candidates = cast(npt.NDArray[np.float64], np.array(new_candidates))
    predictor.train()
    return new_candidates


def gen_predictors(
    X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], n_predictors: int
):
    ...


# %% Main

opt_duration = []

for i_opt in range(n_opts):
    torch.manual_seed(i_opt + 1)

    t_start = time.perf_counter()

    predictors, optimizers = gen_init_predictors(n_predictors, n_dims)

    X = gen_init_X(n_dims, s_init, var_bounds)
    Y = v_cost_fn(X)

    best_ys = [min(Y)]
    idx_best_y = np.argmin(Y)
    best_xs = X[idx_best_y]
    new_candidates, y_new_candidates = X, Y

    for iter in range(n_iters):
        predictors, optimizers, error = update_predictors(
            predictors, optimizers, new_candidates, y_new_candidates
        )

        X_candidates_mut = gen_candidates(X, Y, n_candidates)
        new_candidates = find_new_candidates(
            X_candidates_mut, predictors, n_predictors, n_new_candidates
        )
        y_new_candidates = v_cost_fn(new_candidates)

        X = np.vstack([X, new_candidates])  # pyright: ignore[reportConstantRedefinition]
        Y = np.hstack([Y, y_new_candidates])  # pyright: ignore[reportConstantRedefinition]

        best_ys.append(min(Y))
        idx_best_y = np.argmin(Y)
        best_xs = np.vstack([X, X[idx_best_y]])

        alpha_mutation = (1 - iter / n_iters) * alpha_mut_ratio
        n_dims_mut = int(alpha_mutation * n_dims) + 1

        # with open(
        #     os.path.join(save_path, f"opt_{i_opt}_iter_{iter}.csv"),
        #     mode="a",
        #     newline="",
        # ) as Y_hist_file:
        #     Y_hist_writer = csv.writer(Y_hist_file, lineterminator="\n")
        #     row = [f"{iter + 1}"]
        #     for i in range(n_dims):
        #         row.append("{:.4f}".format(best_xs[-1][i]))
        #     row.append("{:.4f}".format(best_ys[-1]))
        #     Y_hist_writer.writerow(row)
        if iter % 20 == 0:
            logger.info(f"Iter: {iter}, error: {error}")
            logger.info(f"x: {best_xs[-1]}, y: {best_ys[-1]}")

    t_finish = time.perf_counter()
    opt_duration.append(t_finish - t_start)
