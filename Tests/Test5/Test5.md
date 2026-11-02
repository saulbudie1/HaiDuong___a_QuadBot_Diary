# Test Log 5 — Go1 Walk (about 100M steps, local RTX 5060)

**Date:** 2026-10-21
**Result:** Pass for forward walking. The robot tracks an x = 1 command at about 0.96 m/s. Other directions not tested yet.

## Setup

| Setting | Value |
|---|---|
| Task | `Go1JoystickFlatTerrain` (default config, domain randomization on) |
| `ppo_params.num_timesteps` | 100,000,000 |
| `ppo_params.num_evals` | 3 (left over from a short test) |
| `ppo_params.num_resets_per_eval` | 10 (default) |
| Reset every | 1,240 steps (computed, see [Test 9.2](../Test9.2/Test9.2.md)) |
| **Steps actually run** | **101,580,800** (2 epochs of 50,790,400) |
| `num_envs` | 4096 (Colab runs used 8192) |
| Hardware | NVIDIA GeForce RTX 5060 Laptop (8 GB), WSL2 Ubuntu 24.04 |
| Software | JAX 0.12.0, brax 0.14.2, flax 0.12.10, mujoco 3.16.0, warp-lang 1.18.0, playground 0.2.0 |
| Run | `train_test.py` inside `tmux`, log in `run1.log`, checkpoint saved at every evaluation |
| Rollout | `rollout.py` (default env: yaw ±1.2, no pushes), command x = 1, y = 0, yaw = 0, checkpoint `go1_final` |

The step count comes from Brax rounding each evaluation epoch up. The short test (asked 2M, ran 3,276,800) matched this, and so does `ckpt_50790400`.

## Timings

| Stage | Time |
|---|---|
| Compile (JIT) | 0:00:58 |
| Training | 0:18:23 |
| Total | 0:19:22 |
| Speed | about 92,000 steps/s |

For comparison, the Colab T4 runs were about 44,000 steps/s (my estimate from Tests 3 and 4).

## Reward (from `run1.log`)

| Env steps | Reward per episode |
|---|---|
| 0 | 0.001 |
| 50,790,400 | 19.556 |
| 101,580,800 | 26.248 |

Only three points, because `num_evals` was 3. The Colab runs (about 29.5M steps) peaked around 17 to 19.

## Rollout numbers (command x = 1, y = 0, yaw = 0)

| | Before (Test 3, standing) | Test 5 (`go1_final`) |
|---|---|---|
| mean `track` | 0.0187 | **0.9405** |
| mean `linvel` (body frame) | about 0, 0, 0 | **0.961, 0.007, 0.018** |

- `track` is `exp(-error² / 0.25)`, so 0.94 means a small velocity error.
- The robot moves forward at about 0.96 m/s, with almost no sideways or vertical motion.
- The means cover the whole 20 s, so they include the start.

## Videos (20 s each, 3 cameras)

- Leg motion is high (frame-difference score about 5 to 10, vs about 0.2 to 0.5 for the standing policies).
- The floor lighting and shadows change across the clip, and the pose cycles through a stepping gait.
- The side-view contact sheet shows the robot upright and stepping, with no collapse.
- I did not measure step frequency or check for sliding or hopping.

## Compared with earlier tests

| | Test 3 / 4 (Colab) | Test 5 (local) |
|---|---|---|
| Steps run | about 29.5M | about 101.6M |
| Final reward | about 17 to 19 (Test 3), about 9 (Test 4) | 26.2 |
| x = 1 behaviour | Stands still | Walks forward, about 0.96 m/s |

## Takeaway

- The policy walks forward after about 3.4 times more training steps. This fits the idea that the earlier runs were undertrained (the official setup uses 200M steps).
- It is one run. Other things also changed: GPU, software versions, `num_envs` (4096 vs 8192), and the rollout setup. So steps are the likeliest cause, not a proven one.
- Reward alone was not enough to tell. The rollout numbers (`track`, `linvel`) were.

**Next**
1. Test other commands: yaw (0 / 0 / 1), sideways (0 / 0.5 / 0), backward (-0.5 / 0 / 0), mixed (1 / 0.5 / 0.5).
2. Compare checkpoints on x = 1: `ckpt_0`, `ckpt_50790400`, `go1_final`.
3. Control run: same local setup with the default `num_evals` (about 29.5M steps). Does it still stand still?
4. A run with `num_evals` 10 to 20 for a finer reward curve and more checkpoints.
5. Try 200M, and a second seed.

## Evidence

- Training log: `run1.log` (in WSL `~/go1`, not copied here)
- [Side view](./100%20mil%20steps/side_x1.0_y0.0_yaw0.0_go1_final.mp4)
- [Top view](./100%20mil%20steps/top_x1.0_y0.0_yaw0.0_go1_final.mp4)
- [Track view](./100%20mil%20steps/track_x1.0_y0.0_yaw0.0_go1_final.mp4)

## Open questions

- Test 6 (same laptop, about 29.5M steps) stands still, so steps are the main driver in these runs. Still untested: `num_envs` 8192, seed variation.
- How good the gait is (speed variation, sliding, hopping).
- Whether the policy follows yaw, sideways and backward commands.
- Whether the constant solver warnings affect training speed.
- How much results vary between seeds.
