import random
from typing import cast

import benchmark_functions as bf  # pyright: ignore
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

T_VAR_BOUNDS = tuple[list[float], list[float]]


@dataclass
class OptParams:
    n_dims: int = 8
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


def gen_init_vX(s_init: int, var_bounds: T_VAR_BOUNDS) -> npt.NDArray[np.float32]:
    lower_bounds = np.array(var_bounds[0])
    upper_bounds = np.array(var_bounds[1])
    vX = np.array([])
    for _ in range(s_init):
        v = rng.uniform(lower_bounds, upper_bounds)
        vX = np.vstack([vX, v]) if vX.size else v
    return cast(npt.NDArray[np.float32], vX)


def cost_fn(vX: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.apply_along_axis(obj_fn, 1, vX)  # pyright: ignore


def gen_init_predictors(
    num_predictors: int, dim_in: int
) -> tuple[list[torch.nn.Sequential], list[torch.optim.Adam]]:
    predictors = []
    optimizers = []
    for _ in range(num_predictors):
        net = torch.nn.Sequential(
            torch.nn.Linear(dim_in, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, 1),
        )
        predictors.append(net)
        optimizers.append(torch.optim.Adam(net.parameters(), lr=LEARNING_RATE))
    return predictors, optimizers  # pyright: ignore


def update_predictors(
    predictors: list[torch.nn.Sequential],
    optimizers: list[torch.optim.Adam],
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float32],
):
    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(Y)
    loss_func = torch.nn.MSELoss()
    errors = []
    for i in range(len(predictors)):
        net = predictors[i]
        optimizer = optimizers[i]
        # optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        prediction = net(x_tensor.float())
        loss = loss_func(prediction, y_tensor.float().view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        errors.append(loss.detach())
    avg_error = torch.mean(torch.stack(errors))
    return predictors, optimizers, avg_error


def mutate(
    x_candidate: npt.NDArray[np.float32], var_bounds: T_VAR_BOUNDS, n_mutations: int
):
    x_candidate_mut = x_candidate.copy()
    idx = rng.choice(np.arange(len(x_candidate)), n_mutations)
    lower_bounds = np.array(var_bounds[0])[idx]
    upper_bounds = np.array(var_bounds[1])[idx]
    x_candidate_mut[idx] = lower_bounds + rng.random(len(idx)) * (
        upper_bounds - lower_bounds
    )
    return x_candidate_mut


def gen_candidates(
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float32],
    num_candidates: int,
    n_mutations: int,
    var_bounds: T_VAR_BOUNDS,
):  # Step 2
    idx = np.argpartition(Y, num_candidates - 1)
    x_candidates = []
    y_candidates = []
    for i in range(num_candidates):
        x_candidates.append(X[idx[i]])
        y_candidates.append(Y[idx[i]])
    x_candidates = cast(npt.NDArray[np.float32], np.array(x_candidates))
    y_candidates = cast(npt.NDArray[np.float32], np.array(y_candidates))
    lower_bounds = np.amin(x_candidates, axis=0)
    upper_bounds = np.amax(x_candidates, axis=0)
    x_candidates_mut = []
    for candidate in x_candidates:
        x_candidates_mut.append(
            # mutate(candidate, (lower_bounds, upper_bounds), n_mutations)
            mutate(candidate, var_bounds, n_mutations)
        )
    x_candidates_mut = cast(npt.NDArray[np.float32], np.array(x_candidates_mut))
    return x_candidates_mut


def find_new_candidates(
    X_mutated_candidate: npt.NDArray[np.float32],
    predictors: list[torch.nn.Sequential],
    num_predictors: int,
    num_new_candidates: int,
):
    new_candidates = []
    predictor = predictors[rng.integers(0, num_predictors - 1)]
    predictor.eval()
    with torch.no_grad():
        prediction = predictor(torch.from_numpy(np.array(X_mutated_candidate)).float())
        prediction = cast(npt.NDArray[np.float32], prediction.numpy())
    min_idx = (
        np.argpartition(prediction.transpose(), num_new_candidates - 1)
    ).transpose()
    for i in range(num_new_candidates):
        new_candidates.extend(X_mutated_candidate[min_idx[i]])
    new_candidates = cast(npt.NDArray[np.float32], np.array(new_candidates))
    predictor.train()
    return new_candidates


def main():
    global n_mutations

    for i_opt in range(n_opt):
        torch.manual_seed(i_opt + 1)
        random.seed(i_opt + 1)

        predictors, optimizers = gen_init_predictors(num_predictors, n_dimensions)

        X = gen_init_vX(s_init, var_bounds)
        Y = cost_fn(X)

        bestY_history = [min(Y)]
        new_candidates, y_new_candidates = X, Y

        for iter in range(n_iterations):
            predictors, optimizers, error = update_predictors(
                predictors, optimizers, new_candidates, y_new_candidates
            )
            X_mutated_candidate = gen_candidates(
                X, Y, num_candidates, n_mutations, var_bounds
            )  # Create Mutation

            new_candidates = find_new_candidates(
                X_mutated_candidate, predictors, num_predictors, num_new_candidates
            )
            y_new_candidates = cost_fn(new_candidates)

            X = np.vstack([X, new_candidates])  # pyright: ignore
            Y = np.append(Y, y_new_candidates)  # pyright: ignore
            bestY_history.append(min(Y))

            n_mutations = (
                int((1 - iter / n_iterations) * alpha_mutation_var * n_dimensions) + 1
            )

            logger.debug(f"{iter=} | y={min(Y)} | error={error}")


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    n_dimensions = 8
    s_init = 2 * n_dimensions
    num_candidates = 2 * n_dimensions
    num_new_candidates = 2 * n_dimensions
    num_predictors = 5
    n_iterations = 200
    alpha_mutation_var = 0.3
    n_mutations = int(alpha_mutation_var * n_dimensions) + 1
    n_opt = 1

    LEARNING_RATE = 0.01
    MODEL_DIM = 10 * n_dimensions

    obj_fn = bf.Griewank(n_dimensions)
    var_bounds = cast(T_VAR_BOUNDS, obj_fn.suggested_bounds())

    main()
