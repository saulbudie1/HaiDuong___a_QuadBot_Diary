"""Train the Go1 joystick policy on a custom ground (any scene XML), with the same PPO settings as train_test.py.

    python train_terrain.py /path/to/ground_scene.xml --steps 120000000 --evals 16 --envs 4096

Run it inside the folder the checkpoints should go to (ckpt_<step> after every evaluation, go1_final at the end).
Same as train_test.py except for the ground: the scene XML is loaded under Playground's task name "rough_custom" (the name
starts with "rough", so it gets the larger contact buffers the stock rough task gets; see rollout_terrain.py, which loads
the ground the same way). Domain randomization is the stock Go1 one, and the PPO settings are those of Go1JoystickRoughTerrain
(the same network as the flat task), with num_timesteps, num_evals and num_envs overridden from the command line.
Every episode starts within 0.5 m of the origin (Playground's reset), so on a ring ground it starts on the flat middle.
"""
import functools
import sys
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from mujoco_playground import wrapper
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
from mujoco_playground._src.locomotion.go1 import joystick, randomize
from mujoco_playground.config import locomotion_params


# Shim: Brax still calls jax.device_put_replicated, removed in JAX 0.11+.
def _device_put_replicated(tree, devices):
    mesh = Mesh(np.array(devices), ("d",))
    sh = NamedSharding(mesh, P("d"))
    return jax.tree.map(
        lambda a: jax.device_put(jnp.stack([jnp.asarray(a)] * len(devices)), sh), tree)


jax.device_put_replicated = _device_put_replicated

argv = sys.argv[1:]


def opt(name, default, cast):
    if name in argv:
        i = argv.index(name)
        v = cast(argv[i + 1])
        del argv[i:i + 2]
        return v
    return default


steps = opt("--steps", 120_000_000, int)
evals = opt("--evals", 16, int)
envs = opt("--envs", 4096, int)
seed = opt("--seed", 0, int)
# --resume FILE: start from the weights in FILE (a ckpt_<step> or go1_final saved by an earlier run) instead of from scratch.
# Only the weights are restored (the normalizer, the policy and the value network). The optimizer state and the step counter
# start again, so the step numbers printed and used in the ckpt_<step> names count from 0 again, and it is not the same as one
# unbroken run. Used when a long run crashes with a GPU out-of-memory error: run the rest in a fresh process in a new folder.
resume = opt("--resume", None, str)
# --resets N: set ppo_params.num_resets_per_eval (the default of the Go1 config is 10, a forced reset of all robots every 200 steps
# at 120M steps, 4096 envs and 16 evaluations). The interval in steps is printed at the start. N = 2 gives 1000 steps (the
# episode cap, 20 s); N = 0 means no forced reset (episodes end only by a fall or the cap). Without the option nothing changes.
resets = opt("--resets", None, int)
ground = argv[0]

xml = epath.Path(ground)
_stock = consts.task_to_xml
consts.task_to_xml = lambda t: xml if t == "rough_custom" else _stock(t)


def make_env():
    return joystick.Joystick(task="rough_custom", config=joystick.default_config())


env = make_env()
eval_env = make_env()
print("ground:", ground, "| floor geom id:", env.mj_model.geom("floor").id,
      "(the domain randomizer assumes", randomize.FLOOR_GEOM_ID, ")", flush=True)
assert env.mj_model.geom("floor").id == randomize.FLOOR_GEOM_ID

ppo_params = locomotion_params.brax_ppo_config("Go1JoystickRoughTerrain")
ppo_params.num_timesteps = steps
ppo_params.num_evals = evals
ppo_params.num_envs = envs
if resets is not None:
    ppo_params.num_resets_per_eval = resets
print("num_timesteps", steps, "num_evals", evals, "num_envs", envs, "seed", seed, flush=True)
# Reset interval, computed from Brax's ppo/train.py (not read back from the run): updates per reset chunk x steps per robot per update.
_per_update = ppo_params.batch_size * ppo_params.unroll_length * ppo_params.num_minibatches
_chunk = -(-steps // (max(evals - 1, 1) * _per_update * max(ppo_params.num_resets_per_eval, 1)))
if ppo_params.num_resets_per_eval > 0:
    print("num_resets_per_eval", ppo_params.num_resets_per_eval, "-> forced reset of all robots every",
          _chunk * _per_update // envs, "steps (computed); episode_length", ppo_params.episode_length, flush=True)
else:
    print("num_resets_per_eval 0 -> no forced resets; episode_length", ppo_params.episode_length, flush=True)

x_data, y_data = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    print(f"steps={num_steps:>10}  reward={y_data[-1]:.3f}  t={times[-1] - times[0]}", flush=True)


ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )


def save_ckpt(step, make_policy, params):
    model.save_params(f"ckpt_{step}", params)


restore_kwargs = {}
if resume:
    restore_kwargs["restore_params"] = model.load_params(resume)  # (normalizer, policy, value), as saved by save_params
    print("resuming from the weights in", resume, flush=True)

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    **restore_kwargs,
    seed=seed,
    network_factory=network_factory,
    randomization_fn=randomize.domain_randomize,
    progress_fn=progress,
    policy_params_fn=save_ckpt,
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print("time to jit:", times[1] - times[0])
print("time to train:", times[-1] - times[1])
print("last step count:", x_data[-1])
model.save_params("go1_final", params)
