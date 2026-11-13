# Test Log 4.1 — Test 4 rerun on the local RTX 5060 (26,214,400 steps, `num_evals` 3)

**Date:** 2026-11-12 (training started 23:59; the three reruns finished at 00:14 on 2026-11-13)
**Result:** Stands still. `track` 0.0189, the robot stays upright and moves 0.05 m in 20 s, as in Test 4. Reward 18.4 at the end, with no late drop visible because `num_evals` 3 gives only three points. One seed, one rollout.

**Purpose:** Rerun [Test 4](../Test4/Test4.md) (Colab, 25M steps requested, ran about 29.5M; reward peaked at about 17.7 then fell to 9.3) on the local RTX 5060 with `num_evals` 3.

## Setup

| Setting | Value |
|---|---|
| Task | `Go1JoystickFlatTerrain` (default config) |
| `ppo_params.num_timesteps` | 25,000,000 |
| `ppo_params.num_evals` | **3** |
| `ppo_params.num_resets_per_eval` | 10 (default) |
| Reset every | 320 steps (computed: 13,107,200 / 4096 / 10, see [Test 9.2](../Test9.2/Test9.2.md)) |
| **Steps actually run** | **26,214,400** (2 epochs of 13,107,200), against about 29.5M on Colab |
| `num_envs` | 4096 (Colab used 8192) |
| Seed | not set in the script (default) |
| Hardware | NVIDIA GeForce RTX 5060 Laptop (8 GB), WSL2 Ubuntu 24.04 |
| Run | `Scripts/train_test.py` copied to `~/go1_t4_1`, only `num_timesteps` and `num_evals` changed with `sed` ([settings.txt](./settings.txt)). Log: [train.log](./train.log) |
| Rollout | `rollout.py`, command x = 1, y = 0, yaw = 0, checkpoint `go1_final`, 1000 steps (20 s), cameras `top`, `side`, `track` |

The step count matches the formula in the notes (predicted before the run).

## Timings

| Stage | Time |
|---|---|
| Compile (JIT) | 0:00:58 |
| Training | 0:06:24 |

## Reward

| Env steps | Reward per episode |
|---|---|
| 0 | 0.001 |
| 13,107,200 | 16.390 |
| 26,214,400 | **18.407** |

Only three points ([reward_plot.png](./reward_plot.png)). Test 4 on Colab peaked at about 17.7 near 22.8M and fell to about 9.3 at 29.5M. This run stopped at 26.2M and has no point in between, so it cannot show a peak or a drop. The late drop in Test 4 is therefore neither repeated nor ruled out.

## Rollout numbers (command x = 1, y = 0, yaw = 0)

| | Test 4.1 | Test 4 (Colab) |
|---|---|---|
| mean `track` | **0.0189** (max 0.072) | not recorded |
| mean `linvel` (body frame) | 0.002, -0.001, 0.002 | not recorded |
| Distance from start in 20 s | 0.05 m | does not walk |
| Upright (1 = level) | min 1.0, end 1.0 | stays upright |
| Body height | 0.29 to 0.33 m | not recorded |

Files: [reward_plot.png](./reward_plot.png), [motion](./motion_x1.0_y0.0_yaw0.0_go1_final.png), [trail](./trail_x1.0_y0.0_yaw0.0_go1_final.png), three camera videos in [videos/](./videos/), [rollout_numbers.txt](./rollout_numbers.txt). I did not watch the videos. The statement that it stands still rests on the numbers above. Test 4 also reported that the final policy "no longer turns continuously"; that was not checked here, because this rollout uses a forward command only.

## Takeaway

- At about 26M steps the laptop policy stands, like Test 4 and [Test 6](../Test6/Test6.md) (about 29.5M steps). Walking appeared only later in my runs (about 48M to 65M steps for most seeds).
- All three reruns (1.1, 2.1, 4.1) stand still, at 6.5M, 19.7M and 26.2M steps, with reset intervals of 80, 240 and 320 steps. So the reset interval was not the limit here. The step count was.
- A reward of 18.4 is a standing robot. Total reward cannot tell the difference at this level.
- Hardware, software versions, `num_envs`, `num_evals` and the step total all differ from Test 4. One seed and one rollout per test.
