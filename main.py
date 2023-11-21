from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

rng = random.PRNGKey(100)


class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray


def init(rng: jax.Array) -> Params:
    weights_key, bias_key = random.split(rng)
    weight = random.normal(weights_key, ())
    bias = random.normal(bias_key, ())
    return Params(weight, bias)


def loss(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = params.weight * x + params.bias
    return jnp.mean((pred - y) ** 2)


LEARNING_RATE = 0.005


@jax.jit
def update(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> Params:
    grad = jax.grad(loss)(params, x, y)

    new_params = jax.tree_map(lambda param, g: param - g * LEARNING_RATE, params, grad)

    return new_params


true_w, true_b = 2, -1
x_rng, noise_rng = random.split(rng)
xs = random.normal(x_rng, (128, 1))
noise = random.normal(noise_rng, (128, 1)) * 0.5
ys = xs * true_w + true_b + noise

if __name__ == "__main__":
    params = init(rng)
    for _ in range(1000):
        params = update(params, xs, ys)

    plt.scatter(xs, ys)
    plt.scatter(xs, params.weight * xs + params.bias, c="red", label="model prediction")
    plt.legend()
    plt.show()
