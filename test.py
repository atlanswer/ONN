# %%
import chex
import jax
import jax.numpy as jnp
import numpy as np
from clu import metric_writers, periodic_actions

chex.set_n_cpu_devices(8)

logdir = "./metrics"
total_steps = 100

writer = metric_writers.create_default_writer(logdir)

hooks = [
    periodic_actions.ReportProgress(
        num_train_steps=total_steps, every_steps=10, writer=writer
    ),
    periodic_actions.Profile(logdir=logdir),
]

for step in range(total_steps):
    # writer.write_scalars(step, dict(loss=0.9**step))
    for hook in hooks:
        hook(step)

#%%
