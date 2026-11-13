# Test Log 1.1 — Test 1 rerun on the local RTX 5060 (6,553,600 steps, `num_evals` 3)

**Date:** 2026-11-12 (training started 23:32; the three reruns finished at 00:14 on 2026-11-13)
**Result:** Stands still. `track` 0.0185, the robot stays upright and moves 0.03 m in 20 s. The Colab Test 1 robot stepped for about 8 s and then fell, so that result did not repeat on this laptop. One seed, one rollout.

**Purpose:** Rerun [Test 1](../Test1/Test1.md) (Colab T4, 6M steps requested) on the local RTX 5060 with `num_evals` 3, to see whether the Colab result repeats. Tests 2.1 and 4.1 do the same for Tests 2 and 4.

## Setup

| Setting | Value |
|---|---|
| Task | `Go1JoystickFlatTerrain` (default config) |
| `ppo_params.num_timesteps` | 6,000,000 |
| `ppo_params.num_evals` | **3** (Test 1's value is not recorded in its log) |
| `ppo_params.num_resets_per_eval` | 10 (default) |
| Reset every | 80 steps (computed: 3,276,800 / 4096 / 10, see [Test 9.2](../Test9.2/Test9.2.md)) |
| **Steps actually run** | **6,553,600** (2 epochs of 3,276,800), against about 6.3M on Colab |
| `num_envs` | 4096 (Colab used 8192) |
| Seed | not set in the script (default), the same as the earlier runs |
| Hardware | NVIDIA GeForce RTX 5060 Laptop (8 GB), WSL2 Ubuntu 24.04 |
| Run | `Scripts/train_test.py` copied to `~/go1_t1_1`, only `num_timesteps` and `num_evals` changed with `sed` ([settings.txt](./settings.txt)). Log: [train.log](./train.log) |
| Rollout | `rollout.py`, command x = 1, y = 0, yaw = 0, checkpoint `go1_final`, 1000 steps (20 s) |

The step count matches the formula in the notes (predicted before the run).

## Timings

| Stage | Time |
|---|---|
| Compile (JIT) | 0:00:58 |
| Training | 0:03:00 |

## Reward

| Env steps | Reward per episode |
|---|---|
| 0 | 0.001 |
| 3,276,800 | 0.012 |
| 6,553,600 | **4.555** |

With `num_evals` 3 the plot has only three points ([reward_plot.png](./reward_plot.png)), so it shows no shape. Test 1 on Colab read about 0.65 at its peak (about 4.9M steps) and about 0.37 at 6M.

## Rollout numbers (command x = 1, y = 0, yaw = 0)

| | Test 1.1 | Test 1 (Colab) |
|---|---|---|
| mean `track` | **0.0185** (max 0.037) | not recorded |
| mean `linvel` (body frame) | 0.001, -0.002, 0.002 | not recorded |
| Distance from start in 20 s | 0.03 m | not recorded |
| Upright (1 = level) | min 1.0, end 1.0 | upright and stepping to about 8 s, then fell |
| Body height | 0.29 to 0.32 m | not recorded |

`track` of 0.0185 is the standing value (`exp(-4)` is about 0.018, see [Test 6](../Test6/Test6.md)).

Files: [reward_plot.png](./reward_plot.png), [motion](./motion_x1.0_y0.0_yaw0.0_go1_final.png), [trail](./trail_x1.0_y0.0_yaw0.0_go1_final.png), three camera videos in [videos/](./videos/), [rollout_numbers.txt](./rollout_numbers.txt). I did not watch the videos. The statement that it stands still rests on the numbers above.

## Takeaway

- At about 6.5M steps the laptop policy stands. This fits [Test 6](../Test6/Test6.md) (about 29.5M steps, stands) and the walking onset of about 48M to 65M steps found later.
- It differs from Colab Test 1, which fell after 8 s. Several things changed at once (hardware, software versions, `num_envs`, `num_evals`, 6.55M against about 6.3M steps), so I cannot say which one matters.
- One seed and one rollout. The wording should be "in my runs".
