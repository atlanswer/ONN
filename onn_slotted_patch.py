# %% Import
import asyncio
import csv  # pyright: ignore
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import benchmark_functions as bf  # pyright: ignore
import matplotlib.pyplot as plt  # pyright: ignore
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from antcal.model.slotted_patch import (  # pyright: ignore[reportMissingTypeStubs]
    N_DIMS_SLOTTED_PATCH,
    VAR_BOUNDS,
    check_constrains,
    obj_fn,
)
from antcal.pyaedt.hfss import new_hfss_session  # pyright: ignore
from antcal.utils import submit_tasks, refresh_aedt_list  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger
from matplotlib import cm  # pyright: ignore
from pyaedt.hfss import Hfss  # pyright: ignore[reportMissingTypeStubs]

# %%
cwd = os.getcwd()
save_path = os.path.join(cwd, "save")
save_dir = Path(save_path)
save_dir.mkdir(exist_ok=True)
logger.add("log.log")


# %% Init
# Optimization & hyperparameters
@dataclass
class OptParams:
    n_dims: int = N_DIMS_SLOTTED_PATCH
    s_init: int = 2 * N_DIMS_SLOTTED_PATCH
    n_candidates: int = 2 * N_DIMS_SLOTTED_PATCH
    n_new_candidates: int = 2 * N_DIMS_SLOTTED_PATCH
    n_predictors: int = N_DIMS_SLOTTED_PATCH
    n_iters: int = 200
    alpha_mut_ratio: float = 0.5
    n_dims_mut: int = int(alpha_mut_ratio * n_dims) + 1
    n_opt_runs = 1
    LEARNING_RATE: float = 0.01
    N_MODEL_DIMS: int = 100 * n_dims
    N_SIMULATORS = 4


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
        while True:
            v_r = rng.random(opt_params.n_dims)
            v = lower_bounds + v_r * (upper_bounds - lower_bounds)
            if check_constrains(v):
                break
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
    candidate: npt.NDArray[np.float32],
    var_bounds: tuple[list[float], list[float]],
    rng: np.random.Generator,
):
    while True:
        candidate_mut = candidate.copy()
        idx = rng.choice(np.arange(len(candidate)), opt_params.n_dims_mut)
        lower_bounds = np.array(var_bounds[0])[idx]
        upper_bounds = np.array(var_bounds[1])[idx]
        candidate_mut[idx] = lower_bounds + rng.random(len(idx)) * (
            upper_bounds - lower_bounds
        )
        if check_constrains(candidate_mut):
            break
    return candidate_mut


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
    X_candidates_mutated = []
    lower_bounds = np.min(X_candidates, axis=0)
    upper_bounds = np.max(X_candidates, axis=0)
    for candidate in X_candidates:
        X_candidates_mutated.append(
            mutate(opt_params, candidate, (lower_bounds, upper_bounds), rng)
        )
    X_candidates_mutated = cast(npt.NDArray[np.float32], np.array(X_candidates_mutated))
    return X_candidates_mutated


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


def cost_fn(
    opt_params: OptParams, X: npt.NDArray[np.float32], aedt_list: list[Hfss]
) -> npt.NDArray[np.float32]:
    return asyncio.run(
        submit_tasks(obj_fn, X, opt_params.N_SIMULATORS, aedt_list), debug=True
    )


# def cost_fn(opt_params: OptParams, X: npt.NDArray[np.float32], aedt_list: list[Hfss],) -> npt.NDArray[np.float32]:
#     ...


# %% main
def main():
    iter_duration = []

    for _ in range(opt_params.n_opt_runs):
        # torch.manual_seed(i_opt + 1)

        predictors, optimizers = gen_init_predictors(opt_params)

        X = gen_init_X(opt_params, VAR_BOUNDS, rng)
        Y = cost_fn(opt_params, X, aedt_list)

        best_ys = [min(Y)]
        idx_best_y = np.argmin(Y)
        best_xs = X[idx_best_y]
        new_candidates, y_new_candidates = X, Y

        with open(
            os.path.join(save_path, f"{time_str}.csv"),
            mode="a",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            header = [
                "iter",
                "W",
                "L",
                "Wr",
                "Wu",
                "Lr",
                "Pr",
                "Lh",
                "Lv",
                "fx",
                "fy",
                "y",
                "loss",
            ]
            csv_writer.writerow(header)

        for iter in range(opt_params.n_iters):
            if iter != 0 and iter % 3 == 0:
                logger.debug("Refreshing AEDT list...")
                refresh_aedt_list(aedt_list)
                logger.debug("AEDT list refreshed.")

            t_start = time.time()

            predictors, optimizers, error = update_predictors(
                predictors, optimizers, new_candidates, y_new_candidates
            )

            X_candidates_mut = gen_candidates(opt_params, X, Y, VAR_BOUNDS, rng)
            new_candidates = find_new_candidates(
                opt_params, X_candidates_mut, predictors, rng
            )
            y_new_candidates = cost_fn(opt_params, new_candidates, aedt_list)

            X = np.vstack([X, new_candidates])  # pyright: ignore[reportConstantRedefinition]
            Y = np.hstack([Y, y_new_candidates])  # pyright: ignore[reportConstantRedefinition]

            best_ys.append(min(Y))
            idx_best_y = np.argmin(Y)
            best_xs = np.vstack([X, X[idx_best_y]])

            opt_params.alpha_mut_ratio *= 1 - iter / opt_params.n_iters
            opt_params.n_dims_mut = (
                int(opt_params.alpha_mut_ratio * opt_params.n_dims) + 1
            )

            with open(
                os.path.join(save_path, f"{time_str}.csv"),
                mode="a",
                newline="",
            ) as csv_file:
                csv_writer = csv.writer(csv_file)
                x = best_xs[-1]
                y = best_ys[-1]
                row = [iter, *x, y, f"{error}"]
                csv_writer.writerow(row)

            t_finish = time.time()
            t_duration = int(t_finish - t_start)
            logger.debug(
                f"iter: {iter} | y: {y} | error: {error} | duration: {t_duration} s."
            )
            iter_duration.append(t_duration)

        logger.info(f"duration: {iter_duration} (s)")

        # hist_x1 = [x[0] for x in X]
        # hist_x2 = [x[1] for x in X]

    logger.debug("Cleanup AEDT list.")
    for hfss in aedt_list:
        hfss.close_desktop()


if __name__ == "__main__":
    rng = np.random.default_rng()
    opt_params = OptParams()
    time_str = time.strftime(r"%m%d%H%M", time.localtime())
    aedt_list: list[Hfss] = [
        new_hfss_session(non_graphical=True) for _ in range(opt_params.N_SIMULATORS)
    ]
    try:
        main()
    except Exception as e:
        logger.error(f"Exception occurred during main loop: {e}.")
    finally:
        logger.debug("Cleanup AEDT list.")
        for hfss in aedt_list:
            hfss.close_desktop()

    input("Press ENTER to exit")
