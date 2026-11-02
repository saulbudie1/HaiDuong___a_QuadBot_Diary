# Test Log 6 — Control run (about 29.5M steps, local RTX 5060)

**Date:** 2026-10-21
**Result:** Stands still, like the Colab runs. Walking appears with more steps (Test 5).

**Purpose:** Test whether the Colab runs stood still because of too few steps, or because of the Colab environment (T4, software, `num_envs`). Same laptop, software, script and `num_envs` as Test 5, with fewer steps.

**Emotion** Reliefed to know that it wasn't my laptop that made it work.
## Setup

| Setting | Value |
|---|---|
| Task | `Go1JoystickFlatTerrain` (default config, domain randomization on) |
| `ppo_params.num_timesteps` | 18,000,000 |
| `ppo_params.num_evals` | 10 (default) |
| `ppo_params.num_resets_per_eval` | 10 (default) |
| Reset every | 80 steps (computed, see [Test 9.2](../Test9.2/Test9.2.md)) |
| **Steps actually run** | **29,491,200** (9 epochs of 3,276,800) |
| `num_envs` | 4096 |
| Hardware | NVIDIA GeForce RTX 5060 Laptop (8 GB), WSL2 Ubuntu 24.04 |
| Run | `train_test.py` in `~/go1_ctrl`, log `ctrl.log` |
| Rollout | `rollout.py`, command x = 1, y = 0, yaw = 0, checkpoint `go1_final` |

The step count matches the formula exactly (predicted 29,491,200), and matches what the Colab "18M" and "25M" runs reached (about 29.5M).

## Timings

| Stage | Time |
|---|---|
| Compile (JIT) | 0:00:57 |
| Training | 0:07:24 |
| Total | 0:08:21 |
| Speed | about 66,000 steps/s |

Slower than Test 5's about 92,000 steps/s, probably because this run has more evaluations.

## Reward

Final evaluation reward: **19.613** (from the end of the log). I did not read the other `steps=` lines from `ctrl.log`. Colab runs of this length reached about 17 to 19.

## Rollout numbers (command x = 1, y = 0, yaw = 0)

| | Test 6 (about 29.5M steps) | Test 5 (about 101.6M steps) |
|---|---|---|
| mean `track` | **0.0187** | **0.9405** |
| mean `linvel` (body frame) | 0.002, -0.002, 0.002 | 0.961, 0.007, 0.018 |
| Final reward | 19.6 | 26.2 |

`track` of 0.0187 is `exp(-4)`, which is what standing still at x = 1 gives (velocity error 1 m/s). The same value appeared in Test 3 on Colab.

## Videos (20 s, 3 cameras)

- The robot stays upright and stays in the same place.
- The side-view contact sheet (frames 2 s apart) shows the same pose in every frame, and the floor grid and lighting do not change, so it does not travel.
- Motion (frame-difference score) is about 0.2 to 0.3 after the first second, vs about 5 to 10 for the walking policy.
- Files are in `Tests/Test6/18mil steps/`. The folder is named after the setting (`num_timesteps` 18M); the steps actually run were 29,491,200. They are the control clips, made at about 04:00.

## Takeaway

- With the same machine, software, `num_envs` and script, about 29.5M steps stands still and about 101.6M walks forward. Steps are the main driver in these runs.
- The Colab notebook was not broken. It reproduces the same standing-still result.
- Reward around 19 to 20 went with standing, and 26 went with walking. This is two policies, so it is a lead and not a rule.
- This is one seed per run, so the wording should be "in my runs".

## Open questions

- Where between 50.8M and 101.6M steps walking appears.
- Whether `num_evals` (10 here, 3 in Test 5) has any effect on learning. It should not.
- Whether `num_envs` 8192 changes the result.
- How much results differ between seeds.

## Next

1. Roll out `ckpt_50790400` from the Test 5 run (reward 19.556, so I expect it to stand still).
2. Rerun 100M with `num_evals` 20, and plot `track` against steps for several checkpoints.
3. Try 200M steps and a second seed.

## Evidence

- [Side view](./18mil%20steps/side_x1.0_y0.0_yaw0.0_go1_final.mp4)
- [Top view](./18mil%20steps/top_x1.0_y0.0_yaw0.0_go1_final.mp4)
- [Track view](./18mil%20steps/track_x1.0_y0.0_yaw0.0_go1_final.mp4)
- Training log: `ctrl.log` (in WSL `~/go1_ctrl`, not copied here)
