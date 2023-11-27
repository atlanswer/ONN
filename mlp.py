import os
import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from torchvision.datasets import MNIST

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp

Params = list[tuple[jax.Array, jax.Array]]


def numpy_collate(batch):
    return jax.tree_map(np.asanyarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def relu(x: jax.Array):
    return jnp.maximum(0, x)


def random_layer_params(m: int, n: int, rng: jax.Array, scale: float = 1e-2):
    w_key, b_key = jax.random.split(rng)
    return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(
        b_key, (n,)
    )


def init_network_params(sizes: list[int], rng: jax.Array) -> Params:
    keys = jax.random.split(rng, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def predict(params: Params, image: jax.Array):
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b

    return logits - jax.scipy.special.logsumexp(logits)


batched_predict = jax.vmap(predict, in_axes=(None, 0))


def one_hot(x: jax.Array, k: int, dtype: type[jnp.float32] = jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params: Params, images: jax.Array, targets: jax.Array):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params: Params, images: jax.Array, targets: jax.Array):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 8
batch_size = 128
n_targets = 10


@jax.jit
def update(params: Params, x: jax.Array, y: jax.Array):
    grads = jax.grad(loss)(params, x, y)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


if __name__ == "__main__":
    mnist_dataset = MNIST("./tmp/mnist/", download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(
        mnist_dataset, batch_size=batch_size, num_workers=0
    )

    train_images = np.array(mnist_dataset.train_data).reshape(
        len(mnist_dataset.train_data), -1
    )
    train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

    mnist_dataset_test = MNIST("/tmp/mnist/", download=True, train=False)
    test_images = jnp.array(
        mnist_dataset_test.test_data.numpy().reshape(
            len(mnist_dataset_test.test_data), -1
        ),
        dtype=jnp.float32,
    )
    test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

    params = init_network_params(layer_sizes, jax.random.PRNGKey(time.time_ns()))

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator:
            y = one_hot(y, n_targets)
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
