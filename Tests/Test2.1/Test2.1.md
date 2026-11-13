# Test Log 2.1 — Test 2 rerun on the local RTX 5060 (19,660,800 steps, `num_evals` 3)

**Date:** 2026-11-12 (training started 23:44; the three reruns finished at 00:14 on 2026-11-13)
**Result:** Stands still. `track` 0.0189, the robot stays upright and moves 0.05 m in 20 s. Reward reached 17.7, in the range of the clean Colab rerun (Test 3) and not the 3 of Test 2. One seed, one rollout.

**Purpose:** Rerun [Test 2](../Test2/Test2.md) (Colab, 18M steps requested, reward about 3) on the local RTX 5060 with `num_evals` 3. Test 2 was run in the same session as Test 1 with only the changed cells rerun, and its low reward was never explained.

## Setup

| Setting | Value |
|---|---|
| Task | `Go1JoystickFlatTerrain` (default config) |
| `ppo_params.num_timesteps` | 18,000,000 |
| `ppo_params.num_evals` | **3** |
| `ppo_params.num_resets_per_eval` | 10 (default) |
| Reset every | 240 steps (computed: 9,830,400 / 4096 / 10, see [Test 9.2](../Test9.2/Test9.2.md)) |
| **Steps actually run** | **19,660,800** (2 epochs of 9,830,400) |
| `num_envs` | 4096 (Colab used 8192) |
| Seed | not set in the script (default) |
| Hardware | NVIDIA GeForce RTX 5060 Laptop (8 GB), WSL2 Ubuntu 24.04 |
| Run | `Scripts/train_test.py` copied to `~/go1_t2_1`, only `num_timesteps` and `num_evals` changed with `sed` ([settings.txt](./settings.txt)). Log: [train.log](./train.log) |
| Rollout | `rollout.py`, command x = 1, y = 0, yaw = 0, checkpoint `go1_final`, 1000 steps (20 s) |

The step count matches the formula in the notes (predicted before the run). A clean run in a fresh process, not a rerun of changed cells.

## Timings

| Stage | Time |
|---|---|
| Compile (JIT) | 0:00:58 |
| Training | 0:05:28 |

## Reward

| Env steps | Reward per episode |
|---|---|
| 0 | 0.001 |
| 9,830,400 | 12.652 |
| 19,660,800 | **17.714** |

Only three points ([reward_plot.png](./reward_plot.png)). Test 2 on Colab read about 3 at the end. Test 3 (clean rerun, 18M requested, about 29.5M run) read about 17 to 19.

## Rollout numbers (command x = 1, y = 0, yaw = 0)

| | Test 2.1 | Test 2 (Colab) |
|---|---|---|
| mean `track` | **0.0189** (max 0.078) | not recorded |
| mean `linvel` (body frame) | 0.002, 0.000, 0.002 | not recorded |
| Distance from start in 20 s | 0.05 m | barely moves |
| Upright (1 = level) | min 0.996, end 1.0 | upright for 20 s |
| Body height | 0.29 to 0.33 m | not recorded |

Files: [reward_plot.png](./reward_plot.png), [motion](./motion_x1.0_y0.0_yaw0.0_go1_final.png), [trail](./trail_x1.0_y0.0_yaw0.0_go1_final.png), three camera videos in [videos/](./videos/), [rollout_numbers.txt](./rollout_numbers.txt). I did not watch the videos. The statement that it stands still rests on the numbers above.

## Takeaway

- The behaviour matches Test 2 (upright, barely moves) and Test 6 (about 29.5M steps, stands still).
- The reward does not match Test 2. 17.7 is in Test 3's range. This agrees with the earlier guess that Test 2's low reward came from leftover session state, but it does not prove it.
- A reward of 17.7 here is a standing robot (`track` 0.0189). Total reward cannot tell standing from walking at this level, as in [Test 6](../Test6/Test6.md).
- Hardware, software versions, `num_envs`, `num_evals` and the step total all differ from Test 2, so a difference cannot be put down to one of them. One seed and one rollout.
