# %% Import
import csv  # pyright: ignore
import os
import time
from dataclasses import dataclass
from queue import Queue
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
    obj_fn_queue,
)
from antcal.utils import submit_tasks  # pyright: ignore
from loguru import logger
from matplotlib import cm  # pyright: ignore
from antcal.pyaedt.hfss import new_hfss_session  # pyright: ignore[reportMissingTypeStubs]

# %%
cwd = os.getcwd()
save_path = os.path.join(cwd, "save")
logger.add("log.log")


# %% Init
# Optimization & hyperparameters
@dataclass
class OptParams:
    n_dims: int = N_DIMS_SLOTTED_PATCH
    s_init: int = 6
    n_candidates: int = 3
    n_new_candidates: int = 3
    n_predictors: int = 6
    n_iters: int = 150
    alpha_mut_ratio: float = 0.4
    n_dims_mut: int = int(alpha_mut_ratio * n_dims) + 1
    n_opt_runs = 1
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.01
    N_MODEL_DIMS: int = 100 * n_dims
    EPOCHS: int = 2 * int(s_init / BATCH_SIZE) + 1


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
    opt_params: OptParams
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
    # X_min = X_candidates.min()
    # X_max = X_candidates.max()
    X_candidates_mutated = []
    for candidate in X_candidates:
        X_candidates_mutated.append(mutate(opt_params, candidate, var_bounds, rng))
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


# %% AEDT queue
aedt_queue = Queue()  # pyright: ignore
h1 = new_hfss_session()
h2 = new_hfss_session()
h3 = new_hfss_session()
aedt_queue.put(h1)
aedt_queue.put(h2)
aedt_queue.put(h3)

# %% main

if __name__ == "__main__":
    iter_duration = []
    hist_y = []
    hist_loss = []
    hist_x1 = []
    hist_x2 = []

    opt_params = OptParams()
    rng = np.random.default_rng()

    for i_opt in range(opt_params.n_opt_runs):
        # torch.manual_seed(i_opt + 1)

        predictors, optimizers = gen_init_predictors(opt_params)

        X = gen_init_X(opt_params, VAR_BOUNDS, rng)  # pyright: ignore[reportUnboundVariable]
        Y = submit_tasks(aedt_queue, X, obj_fn_queue)  # pyright: ignore[reportUnboundVariable]

        best_ys = [min(Y)]
        idx_best_y = np.argmin(Y)
        best_xs = X[idx_best_y]
        new_candidates, y_new_candidates = X, Y

        # with open(
        #     os.path.join(save_path, f"opt_{i_opt}.csv"),
        #     mode="a",
        #     newline="",
        # ) as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     header = ["iter", "x1", "x2", "y", "loss"]
        #     csv_writer.writerow(header)

        for iter in range(opt_params.n_iters):
            t_start = time.perf_counter()

            predictors, optimizers, error = update_predictors(
                predictors, optimizers, new_candidates, y_new_candidates
            )

            X_candidates_mut = gen_candidates(opt_params, X, Y, VAR_BOUNDS, rng)
            new_candidates = find_new_candidates(
                opt_params, X_candidates_mut, predictors, rng
            )
            y_new_candidates = submit_tasks(aedt_queue, new_candidates, obj_fn_queue)

            X = np.vstack([X, new_candidates])  # pyright: ignore[reportConstantRedefinition]
            Y = np.hstack([Y, y_new_candidates])  # pyright: ignore[reportConstantRedefinition]

            best_ys.append(min(Y))
            idx_best_y = np.argmin(Y)
            best_xs = np.vstack([X, X[idx_best_y]])

            alpha_mutation = (
                1 - iter / opt_params.n_iters
            ) * opt_params.alpha_mut_ratio
            n_dims_mut = int(alpha_mutation * opt_params.n_dims) + 1

            # with open(
            #     os.path.join(save_path, f"opt_{i_opt}.csv"),
            #     mode="a",
            #     newline="",
            # ) as csv_file:
            #     csv_writer = csv.writer(csv_file)
            #     [x1, x2] = best_xs[-1]
            #     y = best_ys[-1]
            #     row = [iter, x1, x2, y, f"{error}"]
            #     csv_writer.writerow(row)

            if iter % 25 == 0:
                logger.info(f"iter: {iter}, error: {error}")
                logger.info(f"x: {best_xs[-1]}, y: {best_ys[-1]}")

            hist_y.append(best_ys[-1])
            hist_loss.append(error.item())

            t_finish = time.perf_counter()
            iter_duration.append(t_finish - t_start)

        logger.info(f"duration: {iter_duration} (s)")

        hist_x1 = [x[0] for x in X]
        hist_x2 = [x[1] for x in X]

# %% Graph y

# plt.style.use(["seaborn-v0_8-paper", "./publication.mplstyle"])

# hist_x = np.arange(1, len(hist_y) + 1)

# fig, ax = plt.subplots(figsize=(3.5, 2))

# ax.plot(
#     hist_x,
#     hist_y,
#     c=cm.Paired(1),
# )

# ax.set_xlabel("Iteration")
# ax.set_ylabel("Cost Function")

# ax.set_xlim(0, n_iters)
# # ax.set_ylim(-40, 0)
# # ax.set_xticks(np.linspace(23, 25, 5, endpoint=True))
# # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1 / 4))
# ax.grid(which="minor", linestyle=":", alpha=0.5)

# # ax.legend(loc="lower left", prop={"math_fontfamily": "stix"})

# fig.savefig("y.svg", bbox_inches="tight")

# %% Graph loss

# plt.style.use(["default", "seaborn-v0_8-paper", "./publication.mplstyle"])

# hist_x = np.arange(1, len(hist_y) + 1)

# fig, ax = plt.subplots(figsize=(3.5, 2))

# ax.plot(
#     hist_x[1:],
#     hist_loss[1:],
#     c=cm.Paired(3),  # pyright: ignore
# )

# ax.set_xlabel("Iteration")
# ax.set_ylabel("Average Loss")

# ax.set_xlim(0, opt_params.n_iters)
# # ax.set_ylim(-40, 0)
# # ax.set_xticks(np.linspace(23, 25, 5, endpoint=True))
# # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1 / 4))
# ax.grid(which="minor", linestyle=":", alpha=0.5)

# # ax.legend(loc="lower left", prop={"math_fontfamily": "stix"})

# # fig.savefig("loss.svg", bbox_inches="tight")
# fig.show()

# %% Plot 2D

# plt.style.use("default")
# plt.style.use(["seaborn-v0_8-paper", "./publication.mplstyle"])
# fig, ax = plt.subplots(figsize=(3.5, 3.5))

# # Make data.
# pX = np.arange(lower_bounds[0], upper_bounds[0], 2)
# pY = np.arange(lower_bounds[1], upper_bounds[1], 2)
# pX, pY = np.meshgrid(pX, pY)
# pZ = np.zeros(pX.shape)
# for i in range(len(pX)):
#     for j in range(len(pX[0])):
#         pZ[i, j] = cost_fn([pX[i, j], pY[i, j]])

# # Plot the surface.
# ax.pcolormesh(pX, pY, pZ, cmap=cm.viridis, linewidth=0, antialiased=True)
# ax.plot(hist_x1, hist_x2, "o", markersize=3, c=cm.Paired(8))

# ax.set_xlim(lower_bounds[0], upper_bounds[0])
# ax.set_ylim(lower_bounds[1], upper_bounds[1])
# # A StrMethodFormatter is used automatically
# # ax.zaxis.set_major_formatter("{x:.02f}")
# # ax.tick_params(axis="x", which="major", pad=-5)
# # ax.tick_params(axis="y", which="major", pad=-3)
# # ax.tick_params(axis="z", which="major", pad=-2)

# # Add a color bar which maps values to colors.
# # fig.colorbar(surf, shrink=0.5, aspect=10)

# fig.savefig("map.pdf", bbox_inches="tight")
