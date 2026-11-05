"""Test 10, step 1: which forward command speeds get a policy up the first ramp? (no video, one process, compiled once)

    python climb_sweep.py <ground.xml> <checkpoint> [--speeds 0.3,0.5,...] [--seconds 80] [--out climb_sweep.csv]

For each forward speed v the policy is rolled out with command v / 0 / 0 for --seconds on the ground (same start and
PRNGKey(0) as rollout_terrain.py) and the table says how far it got. The ramp rings start 10 m from the centre; "on
the first plateau" means the trunk got to d = max(|x|, |y|) >= 18.5 m and 2.0 m or more above the start, upright, with
no termination before that.
"""
import argparse
import csv
import time

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
from mujoco_playground import registry
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
from mujoco_playground._src.locomotion.go1 import joystick
from mujoco_playground.config import locomotion_params

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("ground")
ap.add_argument("checkpoint")
ap.add_argument("--speeds", default="0.3,0.5,0.7,0.9,1.1,1.3,1.5")
ap.add_argument("--seconds", type=float, default=80.0)
ap.add_argument("--out", default="climb_sweep.csv")
args = ap.parse_args()

xml = epath.Path(args.ground)
_stock = consts.task_to_xml
consts.task_to_xml = lambda t: xml if t == "rough_custom" else _stock(t)
env = joystick.Joystick(task="rough_custom", config=joystick.default_config())
ppo_params = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
network = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    **ppo_params.network_factory,
)
make_inference_fn = ppo_networks.make_inference_fn(network)
params = model.load_params(args.checkpoint)
inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
n_steps = int(round(args.seconds / float(env.dt)))
dt = float(env.dt)


@jax.jit
def measure(state):
    q = state.data.qpos
    up = 1 - 2 * (q[4] ** 2 + q[5] ** 2)
    return jp.stack([q[0], q[1], q[2], up, state.done])


rows = []
print(f"{'speed':>6} {'on plateau':>11} {'arrived at':>11} {'farthest d':>11} {'highest z':>10} {'min upright':>12} {'ended':>14}")
for v in (float(s) for s in args.speeds.split(",")):
    t0 = time.time()
    command = jp.array([v, 0.0, 0.0])
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    state.info["command"] = command
    log = []
    for _ in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        state.info["command"] = command
        log.append(measure(state))
    m = np.asarray(jp.stack(log))                      # x, y, z, upright, done per step
    d = np.maximum(np.abs(m[:, 0]), np.abs(m[:, 1]))
    z0 = m[0, 2]
    arrived = np.where((d >= 18.5) & (m[:, 2] - z0 >= 2.0))[0]
    first_arrive = int(arrived[0]) if len(arrived) else None
    horizon = first_arrive if first_arrive is not None else len(m)
    clean = bool(m[:horizon, 3].min() > 0.5 and m[:horizon, 4].max() < 0.5)
    ok = first_arrive is not None and clean
    first_done = np.where(m[:, 4] > 0.5)[0]
    ended = f"fell at {first_done[0] * dt:.0f} s" if len(first_done) else ("standing/stuck" if d[-1] < 18.5 else "still going")
    rows.append(dict(speed=v, on_plateau=ok, arrive_s=None if first_arrive is None else round(first_arrive * dt, 1),
                     farthest_d=round(float(d.max()), 2), highest_z=round(float(m[:, 2].max()), 2),
                     min_upright=round(float(m[:horizon, 3].min()), 3), ended=ended))
    r = rows[-1]
    print(f"{v:6.2f} {str(ok):>11} {str(r['arrive_s']):>11} {r['farthest_d']:11.2f} {r['highest_z']:10.2f} "
          f"{r['min_upright']:12.3f} {ended:>14}   ({time.time() - t0:.0f} s)", flush=True)

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("saved", args.out)
