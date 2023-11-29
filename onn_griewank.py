# %% Import
import csv
import os
import time
from typing import cast

import benchmark_functions as bf
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from opytimark.markers.n_dimensional import (
    Ackley4,
    Griewank,
)
from opytimark.markers.two_dimensional import Easom, Keane  # pyright:ignore

cwd = os.getcwd()
save_path = os.path.join(cwd, "save")
logger.add("log.log")
rng = np.random.default_rng()

# %% Init
# Optimization parameters
n_dims = 2
s_init = 4
n_candidates = 4
n_new_candidates = 4
n_predictors = 5
n_iters = 100
alpha_mut_ratio = 0.4
n_dims_mut = int(alpha_mut_ratio * n_dims) + 1
n_opts = 1

# Hyperparameters

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MODEL_DIMS = 100 * n_dims
EPOCHS = 2 * int(s_init / BATCH_SIZE) + 1

# Cost function
# cost_fn = Griewank(dims=n_dims)
# cost_fn = Ackley4(dims=n_dims)
# cost_fn = Easom(dims=n_dims)
# cost_fn = Keane(dims=n_dims)
cost_fn = bf.Schwefel(n_dims)
# var_bounds = [(-600.0, 600.0)] * n_dims
# var_bounds = [(-35.0, 35.0)] * n_dims
# var_bounds = [(-5.0, 5.0)] * n_dims
# var_bounds = [(-100.0, 100.0)] * n_dims
# var_bounds = [(0.0, 10.0)] * n_dims
[lower_bounds, upper_bounds] = cost_fn.suggested_bounds()
var_bounds = list(zip(lower_bounds, upper_bounds))

# %% Plot 3D

plt.style.use("default")
plt.style.use(["seaborn-v0_8-paper", "./publication.mplstyle"])
fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"projection": "3d"})

# Make data.
pX = np.arange(lower_bounds[0], upper_bounds[0], 1.0)
pY = np.arange(lower_bounds[1], upper_bounds[1], 1.0)
pX, pY = np.meshgrid(pX, pY)
pZ = np.zeros(pX.shape)
for i in range(len(pX)):
    for j in range(len(pX[0])):
        pZ[i, j] = cost_fn([pX[i, j], pY[i, j]])

# Plot the surface.
surf = ax.plot_surface(pX, pY, pZ, cmap=cm.viridis, linewidth=0, antialiased=True)

# Customize the z axis.
# ax.set_zlim(0, 0)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter("{x:.02f}")
ax.tick_params(axis="x", which="major", pad=-5)
ax.tick_params(axis="y", which="major", pad=-3)
ax.tick_params(axis="z", which="major", pad=-2)
ax.view_init(elev=15, azim=45, roll=0)

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)

fig.savefig("fig.pdf", bbox_inches="tight")

# %% Functions


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
    lower_bounds = np.array([i[0] for i in var_bounds])[idx]
    upper_bounds = np.array([i[1] for i in var_bounds])[idx]
    # for i in range(n_dims_mut):
    # candidate_mut[idx[i]] = global_min + rng.random() * (global_max - global_min)
    candidate_mut[idx] = lower_bounds + rng.random(len(idx)) * (
        upper_bounds - lower_bounds
    )
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
hist_y = []
hist_loss = []
hist_x1 = []
hist_x2 = []

for i_opt in range(n_opts):
    # torch.manual_seed(i_opt + 1)

    t_start = time.perf_counter()

    predictors, optimizers = gen_init_predictors(n_predictors, n_dims)

    X = gen_init_X(n_dims, s_init, var_bounds)
    Y = v_cost_fn(X)

    best_ys = [min(Y)]
    idx_best_y = np.argmin(Y)
    best_xs = X[idx_best_y]
    new_candidates, y_new_candidates = X, Y

    with open(
        os.path.join(save_path, f"opt_{i_opt}.csv"),
        mode="a",
        newline="",
    ) as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["iter", "x1", "x2", "y", "loss"]
        csv_writer.writerow(header)

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

        with open(
            os.path.join(save_path, f"opt_{i_opt}.csv"),
            mode="a",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            [x1, x2] = best_xs[-1]
            y = best_ys[-1]
            row = [iter, x1, x2, y, f"{error}"]
            csv_writer.writerow(row)

        if iter % 25 == 0:
            logger.info(f"Iter: {iter}, error: {error}")
            logger.info(f"x: {best_xs[-1]}, y: {best_ys[-1]}")

        hist_y.append(y)
        hist_loss.append(error.item())

    t_finish = time.perf_counter()
    opt_duration.append(t_finish - t_start)

    hist_x1 = [x[0] for x in X]
    hist_x2 = [x[1] for x in X]

# %% Graph

plt.style.use(["seaborn-v0_8-paper", "./publication.mplstyle"])

hist_x = np.arange(1, len(hist_y) + 1)

fig, ax = plt.subplots(figsize=(3.5, 2))

ax.plot(
    hist_x,
    hist_y,
    c=cm.Paired(1),
)

ax.set_xlabel("Iteration")
ax.set_ylabel("Cost Function")

ax.set_xlim(0, n_iters)
# ax.set_ylim(-40, 0)
# ax.set_xticks(np.linspace(23, 25, 5, endpoint=True))
# ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1 / 4))
ax.grid(which="minor", linestyle=":", alpha=0.5)

# ax.legend(loc="lower left", prop={"math_fontfamily": "stix"})

fig.savefig("y.svg", bbox_inches="tight")

# %%

plt.style.use(["seaborn-v0_8-paper", "./publication.mplstyle"])

hist_x = np.arange(1, len(hist_y) + 1)

fig, ax = plt.subplots(figsize=(3.5, 2))

ax.plot(
    hist_x[1:],
    hist_loss[1:],
    c=cm.Paired(3),
)

ax.set_xlabel("Iteration")
ax.set_ylabel("Average Loss")

ax.set_xlim(0, n_iters)
# ax.set_ylim(-40, 0)
# ax.set_xticks(np.linspace(23, 25, 5, endpoint=True))
# ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1 / 4))
ax.grid(which="minor", linestyle=":", alpha=0.5)

# ax.legend(loc="lower left", prop={"math_fontfamily": "stix"})

fig.savefig("loss.svg", bbox_inches="tight")

# %% Plot 2D

plt.style.use("default")
plt.style.use(["seaborn-v0_8-paper", "./publication.mplstyle"])
fig, ax = plt.subplots(figsize=(3.5, 3.5))

# Make data.
pX = np.arange(lower_bounds[0], upper_bounds[0], 2)
pY = np.arange(lower_bounds[1], upper_bounds[1], 2)
pX, pY = np.meshgrid(pX, pY)
pZ = np.zeros(pX.shape)
for i in range(len(pX)):
    for j in range(len(pX[0])):
        pZ[i, j] = cost_fn([pX[i, j], pY[i, j]])

# Plot the surface.
ax.pcolormesh(pX, pY, pZ, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.plot(hist_x1, hist_x2, "o", markersize=3, c=cm.Paired(8))

ax.set_xlim(lower_bounds[0], upper_bounds[0])
ax.set_ylim(lower_bounds[1], upper_bounds[1])
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter("{x:.02f}")
# ax.tick_params(axis="x", which="major", pad=-5)
# ax.tick_params(axis="y", which="major", pad=-3)
# ax.tick_params(axis="z", which="major", pad=-2)

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)

fig.savefig("map.pdf", bbox_inches="tight")

# %%
