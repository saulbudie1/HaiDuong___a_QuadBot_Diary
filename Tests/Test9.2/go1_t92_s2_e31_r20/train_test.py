import functools
from datetime import datetime
from brax.io import model

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

# Shim: Brax still calls jax.device_put_replicated, removed in JAX 0.11+.
def _device_put_replicated(tree, devices):
    mesh = Mesh(np.array(devices), ("d",))
    sh = NamedSharding(mesh, P("d"))
    return jax.tree.map(
        lambda a: jax.device_put(jnp.stack([jnp.asarray(a)] * len(devices)), sh), tree)

jax.device_put_replicated = _device_put_replicated

env_name = "Go1JoystickFlatTerrain"
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

ppo_params = locomotion_params.brax_ppo_config(env_name)
ppo_params.num_timesteps = 96_000_000
ppo_params.num_evals = 31
ppo_params.num_envs = 4096            # smaller, to fit 8 GB
ppo_params.num_resets_per_eval = 20

x_data, y_data = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    print(f"steps={num_steps:>10}  reward={y_data[-1]:.3f}  t={times[-1] - times[0]}", flush=True)

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
def save_ckpt(step, make_policy, params):
    model.save_params(f"ckpt_{step}", params)

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    policy_params_fn=save_ckpt,
    seed=2,
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print("time to jit:", times[1] - times[0])
print("time to train:", times[-1] - times[1])
print("last step count:", x_data[-1])
model.save_params("go1_final", params)
