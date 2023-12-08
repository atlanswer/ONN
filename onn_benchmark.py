# %% Import
# import asyncio
import csv  # pyright: ignore
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import benchmark_functions as bf  # pyright: ignore
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

# %%
cwd = os.getcwd()
save_path = os.path.join(cwd, "save_synthetic")
save_dir = Path(save_path)
save_dir.mkdir(exist_ok=True)
logger.add("log.log")


# %% Init
# Optimization & hyperparameters
@dataclass
class OptParams:
    n_dims: int = 2
    s_init: int = 2 * n_dims
    n_candidates: int = 2 * n_dims
    n_new_candidates: int = 2 * n_dims
    n_predictors: int = 5
    n_iters: int = 200
    alpha_mut_ratio: float = 0.3
    n_dims_mut: int = int(alpha_mut_ratio * n_dims) + 1
    n_opt_runs = 1
    LEARNING_RATE: float = 0.01
    N_MODEL_DIMS: int = 10 * n_dims


# %% Functions
def gen_init_X(
    opt_params: OptParams,
    var_bounds: tuple[list[float], list[float]],
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    lower_bounds = np.array(var_bounds[0])
    upper_bounds = np.array(var_bounds[1])
    vX = np.array([])
    for _ in range(opt_params.s_init):
        v_r = rng.random(opt_params.n_dims)
        v = lower_bounds + v_r * (upper_bounds - lower_bounds)
        vX = np.vstack([vX, v]) if vX.size else v
    return vX


def gen_init_predictors(
    opt_params: OptParams,
) -> tuple[list[nn.Sequential], list[optim.Adam]]:
    predictors = []
    optimizers = []
    for _ in range(opt_params.n_predictors):
        net = nn.Sequential(
            nn.Linear(opt_params.n_dims, opt_params.N_MODEL_DIMS),
            nn.LeakyReLU(),
            nn.Linear(opt_params.N_MODEL_DIMS, opt_params.N_MODEL_DIMS),
            nn.LeakyReLU(),
            nn.Linear(opt_params.N_MODEL_DIMS, 1),
        )
        predictors.append(net)
        optimizers.append(optim.Adam(net.parameters(), lr=opt_params.LEARNING_RATE))
    return predictors, optimizers  # pyright: ignore[reportUnknownVariableType]


def update_predictors(
    predictors: list[nn.Sequential],
    optimizers: list[optim.Adam],
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float32],
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
    opt_params: OptParams,
    x_candidate: npt.NDArray[np.float32],
    var_bounds: tuple[list[float], list[float]],
    rng: np.random.Generator,
):
    x_candidate_mut = x_candidate.copy()
    idx = rng.choice(np.arange(len(x_candidate)), opt_params.n_dims_mut)
    lower_bounds = np.array(var_bounds[0])[idx]
    upper_bounds = np.array(var_bounds[1])[idx]
    x_candidate_mut[idx] = lower_bounds + rng.random(len(idx)) * (
        upper_bounds - lower_bounds
    )
    return x_candidate_mut


def gen_candidates(
    opt_params: OptParams,
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float32],
    var_bounds: tuple[list[float], list[float]],
    rng: np.random.Generator,
):
    idx = np.argpartition(Y, opt_params.n_candidates - 1)
    X_candidates = []
    Y_candidates = []
    for i in range(opt_params.n_candidates):
        X_candidates.append(X[idx[i]])
        Y_candidates.append(Y[idx[i]])
    X_candidates = cast(npt.NDArray[np.float32], np.array(X_candidates))
    Y_candidates = cast(npt.NDArray[np.float32], np.array(Y_candidates))
    X_candidates_mut = []
    lower_bounds = np.min(X_candidates, axis=0)
    # lower_bounds = [X_candidates.min()] * opt_params.n_dims
    upper_bounds = np.max(X_candidates, axis=0)
    # upper_bounds = [X_candidates.max()] * opt_params.n_dims
    for x_candidate in X_candidates:
        # X_candidates_mut.append(mutate(opt_params, x_candidate, var_bounds, rng))
        X_candidates_mut.append(
            mutate(opt_params, x_candidate, (lower_bounds, upper_bounds), rng)
        )
    X_candidates_mut = cast(npt.NDArray[np.float32], np.array(X_candidates_mut))
    return X_candidates_mut


def find_new_candidates(
    opt_params: OptParams,
    X_candidates_mut: npt.NDArray[np.float32],
    predictors: list[nn.Sequential],
    rng: np.random.Generator,
):
    new_candidates = []
    predictor = predictors[rng.integers(0, opt_params.n_predictors - 1)]
    predictor.eval()
    with torch.no_grad():
        prediction = predictor(torch.from_numpy(np.array(X_candidates_mut)).float())
        prediction = cast(npt.NDArray[np.float32], prediction.numpy())
    min_idx = (
        np.argpartition(prediction.transpose(), opt_params.n_new_candidates - 1)
    ).transpose()
    for i in range(opt_params.n_new_candidates):
        new_candidates.extend(X_candidates_mut[min_idx[i]])
    new_candidates = cast(npt.NDArray[np.float32], np.array(new_candidates))
    predictor.train()
    return new_candidates


# %% main
def main():
    for _ in range(opt_params.n_opt_runs):
        torch.manual_seed(1)

        predictors, optimizers = gen_init_predictors(opt_params)

        X = gen_init_X(opt_params, VAR_BOUNDS, rng)
        Y = cost_fn(opt_params, X)

        best_ys = [min(Y)]
        idx_best_y = np.where(Y == min(Y))
        best_xs = X[idx_best_y[0]]
        new_candidates, y_new_candidates = X, Y

        with open(
            os.path.join(save_path, f"{time_str}.csv"),
            mode="a",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            header = [
                "iter",
            ]
            header.extend([f"x{i}" for i in range(1, opt_params.n_dims + 1)])
            header.extend(
                [
                    "y",
                    "loss",
                ]
            )
            csv_writer.writerow(header)

        for iter in range(opt_params.n_iters):
            t_start = time.time()

            predictors, optimizers, error = update_predictors(
                predictors, optimizers, new_candidates, y_new_candidates
            )

            X_candidates_mut = gen_candidates(opt_params, X, Y, VAR_BOUNDS, rng)
            new_candidates = find_new_candidates(
                opt_params, X_candidates_mut, predictors, rng
            )
            y_new_candidates = cost_fn(opt_params, new_candidates)

            X = np.vstack([X, new_candidates])  # pyright: ignore[reportConstantRedefinition]
            Y = np.append(Y, y_new_candidates)  # pyright: ignore[reportConstantRedefinition]

            best_ys.append(min(Y))
            # idx_best_y = np.argmin(Y)
            idx_best_y = np.where(Y == min(Y))
            best_xs = np.vstack([X, X[idx_best_y[0]]])

            opt_params.n_dims_mut = (
                int(
                    (1 - iter / opt_params.n_iters)
                    * opt_params.alpha_mut_ratio
                    * opt_params.n_dims
                )
                + 1
            )

            with open(
                os.path.join(save_path, f"{time_str}.csv"),
                mode="a",
                newline="",
            ) as csv_file:
                csv_writer = csv.writer(csv_file)
                best_x = best_xs[-1]
                best_y = best_ys[-1]
                row = [iter, *best_x, best_y, f"{error}"]
                csv_writer.writerow(row)

            t_finish = time.time()
            t_duration = int(t_finish - t_start)
            logger.debug(
                f"iter: {iter} | y: {best_y} | error: {error} | duration: {t_duration} s."
            )

        logger.info(obj_fn.testLocalMinimum(best_x))  # pyright: ignore

        ms = obj_fn.minima()  # pyright: ignore
        assert ms is not None
        for m in ms:  # pyright: ignore
            logger.debug(m)


if __name__ == "__main__":
    rng = np.random.default_rng()
    opt_params = OptParams()
    time_str = time.strftime(r"%m%d%H%M", time.localtime())
    obj_fn = bf.Schwefel(opt_params.n_dims)
    VAR_BOUNDS = cast(tuple[list[float], list[float]], obj_fn.suggested_bounds())

    def cost_fn(
        opt_params: OptParams, X: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return np.apply_along_axis(obj_fn, 1, X)  # pyright: ignore

    try:
        main()
    except Exception as e:
        logger.error(f"Exception occurred during main loop: {e}.")
    finally:
        ...

    input("Press ENTER to exit")
