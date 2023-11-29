import chex
import jax

chex.set_n_cpu_devices(8)

print(f"{jax.devices()}")
print(f"{jax.device_count()}")
