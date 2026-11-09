# Testing and Findings

Diary of the Go1 training tests. Times are taken from the screenshot and video file timestamps (UTC+8), so they are approximate.

> **Minimum for walking (so far, flat ground, my runs):** a reset every **80 steps or more** (every 40 steps stands still; only multiples of 40 can be set), and about **50M to 70M training steps** before the switch to walking (earliest about 48M, seed 3 after 68.8M), so I plan for 100M or more. Both are needed: at a 40-step interval the runs stood still even at 96M. Evidence: Test 9.2 (three seeds, one rollout per checkpoint; the reset interval is worked out from my reading of the Brax code, not measured).

---

## 2026-10-20

**Goal:** Train a Go1 policy (`Go1JoystickFlatTerrain`, MuJoCo Playground, Brax PPO on a Colab T4) that walks on command.

**Summary of the day**

| Test | Steps (`num_timesteps`) | Final reward (plot) | Robot behaviour |
|---|---|---|---|
| 1 | 6M | about 0.4 | Steps for about 8 s, then falls |
| 2 | 18M, reran changed cells only | about 3 | Stays up, stands in place |
| 3 | 18M, clean full rerun | about 17.8 | Stays up, turns in place, ignores x/y |
| 4 | 25M (plot to about 29.5M) | about 9.3 | Stays up, turns briefly, then stands still |

Overall: the robot never walked in any direction. Reward is not the same as walking.

---

### 16:05 — Test 1: 6M steps

**Setup:** `num_timesteps` = 6,000,000. Command x = 1, y = 0, yaw = 0. Compile 5:15, training 7:10.

**Code:** Colab setup needed three fixes before Train would run:
```python
!pip install -q warp-lang
!pip install -q -U flax      # then Runtime > Restart session
# plus a shim cell restoring jax.device_put_replicated (Brax 0.14.2 still calls it)
```

**Result:** Reward about 0 until about 3.3M, peak about 0.65 near 4.9M, about 0.37 near 6M. The 20 s video shows the robot stepping for about 8 s, then falling.

**Takeaway:** 6M steps is too short. Log: `Tests/Test1/Test1.md`.

---

### 16:44 — Test 2: 18M steps, reran changed cells only

**Setup:** `num_timesteps` = 18,000,000, same session as Test 1 (no restart). Compile 0:59, training 11:38.

**Code:** I changed only the `num_timesteps` cell and reran Train. I also left `x_vel = 25` in the rollout cell by mistake. The trained range is about ±1.5 m/s, so 25 means nothing. Corrected to `x_vel = 1`.

**Result:** Reward rose to about 3, then flattened. The robot stays upright for 20 s but stands in place (floor grid shifts by under 1 px).

**Takeaway:** More steps fixed the falling, not the walking. The low reward turned out to be unusual, see Test 3. Log: `Tests/Test2/Test2.md`.

---

### 17:28 — Test 3: 18M steps, clean full rerun

**Setup:** Restarted and reran everything with `num_timesteps` = 18M.

**Problem 1 (rendering crash):** Rollout and Render failed with "X11: The DISPLAY environment variable is missing". The fix is to set this before `mujoco` is imported:
```python
import os
os.environ["MUJOCO_GL"] = "egl"
```
**Problem 2 (session crash):** The session crashed once for an unknown reason, so I retrained. The rerun gave reward about 17 to 19 (compile 1:11, training 11:17).

**Rollouts (20 s each):**

| Command (x / y / yaw) | Result |
|---|---|
| 1 / 0 / 0, 1.49 / 0 / 0, 0.3 / 0 / 0, -0.5 / 0 / 0 | Stands in place |
| 0 / 0.8 / 0, 1 / 1 / 0 | Stands in place |
| 0 / 0 / 1 | Steps and turns in place (about 0.9 rad/s, rough) |
| 1 / -1 / 1 | Spins in place, never travels |

**Takeaway:** The command reaches the policy, since yaw works. The policy learned to stay up and to turn, but ignores x and y. Reward about 5 to 6 times higher than Test 2 with the same step count, so Test 2 was probably affected by leftover session state (a guess). Log: `Tests/Test3/Test3.md`.

---

### 22:21 — Test 4: 25M steps, 3 camera views

**Setup:** `num_timesteps` = 25,000,000. The first attempt hit an old-Flax error (`jax.core.get_opaque_trace_state` removed in JAX 0.11), fixed by upgrading Flax and restarting the session. Compile 1:14, training 11:15.

**Code:** The rollout is simulated once, then rendered from three cameras (the Go1 model defines `track`, `top`, `side`, `back`):
```python
for cam in ["top", "side", "track"]:
    frames = eval_env.render(
        traj, camera=cam, scene_option=scene_option,
        width=640, height=480, modify_scene_fns=mod_fns,
    )
    media.show_video(frames, fps=fps, loop=False, title=cam)
    media.write_video(f"{cam}_x{x_vel}_y{y_vel}_yaw{yaw_vel}.mp4", frames, fps=fps)
    del frames
```

**Result:** Reward rose to about 17.7 near 22.8M, then fell to about 9.3 by 29.5M. The plot matches the Test 3 curve until about 22.5M, so training is repeatable. The rollout uses the final (worse) policy.

| Command (x / y / yaw) | Result |
|---|---|
| 0 / 0 / 1 | Turns for the first 2 to 3 s, then holds still |
| 1 / -1 / 1, 1 / 0 / 1 | Settles in about 1 s, then stands still |

No clip shows travel. `1 0 1/side.mp4` and `track.mp4` are the same file (copy mistake), so the track clip wasn't analysed.

**Takeaway:** Training past about 23M made this setup worse. A 40M run with the same settings is unlikely to help. Log: `Tests/Test4/Test4.md`.

---

### Evening — Diagnosis: what the config says

**Code (printed in Colab):**
```python
print(ppo_params)
print("----------\n")
print(env_cfg.command_config)
```
```
!grep -n "def sample_command" -A 15 /usr/local/lib/python3.13/dist-packages/mujoco_playground/_src/locomotion/go1/joystick.py
```

**Findings**
- `command_config`: `a = [1.5, 0.8, 6.283]`, `b = [0.9, 0.25, 0.5]`. Each command component is uniform in ±a, kept with probability `b`, and replaced with probability 0.5 per resample.
- So x is nonzero 90% of the time and was well covered in training. Forward walking was **not** missing from the training commands.
- Yaw goes up to ±6.28 rad/s, not the ±1.2 I had noted earlier. Most yaw commands are far above what the robot can track.
- `ppo_params` look standard (batch 256, unroll 20, 32 minibatches, lr 3e-4, entropy 0.01, discount 0.97, `num_evals` 10).
- Step totals on the plots are about `(num_evals - 1) × 3.28M` (likely, from memory of Brax's formula, unverified). That would explain the plot reaching about 22.5M for 18M and about 29.5M for 25M.

**Hypothesis (untested):** The very wide yaw range may be why the policy learned turning but not forward speed.

---

## Takeaways

1. **Reward is not walking.** Reward went from about 0.4 to about 17.8, yet the robot never travels.
2. **Training is repeatable.** The same setup gave the same curve, so random noise doesn't explain the Test 2 gap. Something about how Test 2 was run differed.
3. **More steps can hurt.** In Test 4, reward peaked near 23M and then declined, and turning stopped being continuous.
4. **The policy follows yaw but not x or y,** across 7 command combinations.
5. **Render the same rollout from every camera.** The same rollout from three cameras makes views comparable, and separate renders can differ.

## Still unknown

- Why the policy ignores x and y.
- Why reward declines after about 23M steps.
- Why training time stayed about 11:15 for both the 18M and 25M runs.
- Whether the rollout policy is deterministic, and whether pushes are active.
- Test 3's `num_evals` (I think 8).
- Which GPU each run used.

## Next

1. **Test 5:** narrow yaw (`a[2]` from 6.28 to about 1.2), keep everything else as in Test 3, and compare.
2. Save intermediate checkpoints so the peak policy (about 23M) can be rendered.
3. Print `linvel` and `track` from the rollout to measure forward motion directly.
4. Read the reward code and explain two terms in my own words.
5. If it still doesn't walk, compare with the unmodified official notebook.

---

## 2026-10-21

**Goal:** Run the long (100M step) training. Colab ran out of compute, so I moved training to my own laptop GPU.

Times are from terminal and screenshot timestamps (UTC+8), approximate.

### Late night (Oct 20) — Colab compute ran out

- I started the 100M run on Colab, and the session stopped because the free compute ran out. I have no results from that run.
- I looked at the paid options: Pay As You Go was SGD 14.46 for 100 compute units (no subscription, units expire after 90 days). I decided to try my own GPU first.

### Notebook and config findings (read from the notebook and printed values)

- **Training environment used the defaults:** printed `train env a = [1.5, 0.8, 1.2]`, pushes off.
- **The yaw 2π and pushes are only in the eval environment**, set by the notebook's "Enable perturbation in the eval env" cell. The rollout cell sets `velocity_kick_range = [0.0, 0.0]`, so rollouts have no pushes.
- **The rollout policy is deterministic** (`make_inference_fn(params, deterministic=True)`).
- **Plot axis:** the notebook plots `xlim = num_timesteps * 1.25`, so the plot can cut off data.
- **Reward arithmetic (from `joystick.py`):** the most a step earns is about `(1.0 + 0.5 + 0.5) × 0.02`, so about 40 per episode. The `pose` term alone is worth about 10, so reward 17 does not need walking.
- **Measured:** for x = 1, y = 0, yaw = 1, mean `track` was 0.0187 and mean `linvel` about 0. `exp(-4) ≈ 0.0183`, so the robot's forward speed is essentially zero.

**Corrections to the Oct 20 entries**
- The idea that the wide yaw range (±6.28) caused the turning-only behaviour was wrong. Training used ±1.2.
- "More steps hurt" (Test 4) is doubtful. The plot cutoff means the 18M and 25M runs probably both ran about 29.5M steps, and their final rewards differ (about 19 vs about 9.3), so the late drop is probably run-to-run variation. Not confirmed.
- Test 2's low reward was probably an evaluation artifact. The eval environment likely had the yaw 2π and pushes, because the rollout cell ran before Train without a restart. Not confirmed for that session.
- My leading explanation is now: the policy is undertrained. The official config uses 200M steps and these runs reached about 29.5M.

### ~00:55 — Local setup (WSL2 + Ubuntu)

| Item | Result |
|---|---|
| Laptop GPU | NVIDIA GeForce RTX 5060 Laptop, 8 GB |
| RAM | about 31 GB (WSL gets about 15 GB) |
| WSL | Installed WSL2 and Ubuntu 24.04 (`wsl --install -d Ubuntu-24.04`) |
| GPU in WSL | `nvidia-smi` shows the GPU, driver 591.84 |
| JAX | 0.12.0 with `jax[cuda12]`, `[CudaDevice(id=0)]`, 4000×4000 matmul printed `4000.0` |
| Warp | 1.18.0, sees `cuda:0` (sm_120) |
| Others | mujoco 3.16.0, brax 0.14.2, flax 0.12.10, playground 0.2.0 |
| Environment | `Go1JoystickFlatTerrain` loaded, action size 12 |

**Code:**
```
python3 -m venv ~/go1env
source ~/go1env/bin/activate
pip install -U "jax[cuda12]"
pip install -U mujoco mujoco-mjx brax warp-lang flax mediapy playground
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

A red "failed to allocate 4.35GiB" message appeared in the first GPU test. It was JAX's memory preallocation, and the calculation still ran.

### ~01:00 — First local test run (`train_test.py`)

The script is the notebook's Train cell trimmed for the terminal (shim for `jax.device_put_replicated`, domain randomization kept, `progress` prints instead of plotting).

| Setting | Value |
|---|---|
| `num_timesteps` (requested) | 2,000,000 |
| `num_evals` | 3 |
| `num_envs` | 4096 |
| Steps actually run | 3,276,800 |
| Reward at 3.28M | 0.015 |
| Time to jit | 2:24 |
| Time to train | 2:23 (about 23,000 steps/s) |

The solver and linesearch iteration warnings printed constantly, but the run completed.

**Takeaway:** Training works on the laptop GPU. At about 23k steps/s, 100M steps would take roughly 70 minutes (one short measurement, not reliable).

### 02:21 — First local 100M run

| Setting | Value |
|---|---|
| `num_timesteps` | 100,000,000 |
| `num_envs` | 4096 (not 8192, to fit 8 GB; Colab used 8192) |
| `num_evals` | default (10) |
| Saving | checkpoint at every evaluation (`ckpt_<step>`) and `go1_final` at the end |
| How it runs | inside `tmux` (session `train`), laptop plugged in, log in `run1.log` |

**Code added to the script:**
```python
from brax.io import model

def save_ckpt(step, make_policy, params):
    model.save_params(f"ckpt_{step}", params)

# in the train_fn partial: policy_params_fn=save_ckpt,
# at the end of the file:
model.save_params("go1_final", params)
```
```
python train_test.py 2>&1 | grep -v -E "iterations limit|To disable the print warning" | tee run1.log
```

At 02:33 the run was in progress (GPU working, laptop fans loud, 74°C, 60% GPU use, 1.3 GB memory). The `num_evals = 3` line from the short test was still in the script, so the run was 2 epochs of 50.8M steps.

### ~02:50 — Run finished

| Env steps | Reward |
|---|---|
| 0 | 0.001 |
| 50,790,400 | 19.556 |
| 101,580,800 | **26.248** |

- **Steps actually run: 101,580,800.**
- Time to jit 0:00:58, time to train 0:18:23, total 0:19:22, about **92,000 steps/s** (about 2x my estimate for the Colab T4).
- Checkpoints saved: `ckpt_0`, `ckpt_50790400`, `ckpt_101580800`, `go1_final`. Backed up to `~/backup_run1`.
- **Step formula (now confirmed twice):** each training step is `batch_size × unroll_length × num_minibatches` = 163,840 env steps. Each epoch is `ceil(num_timesteps / ((num_evals - 1) × 163,840 × 10))` × 10 × 163,840. The test run (asked 2M, ran 3,276,800) and `ckpt_50790400` both match. So the Colab "18M" and "25M" runs (with `num_evals` 10) each ran about 29.5M steps.

### ~03:00 — Test 5: rollout of `go1_final`

**Code (`rollout.py`):** rebuilds the policy network the way training did, loads the saved weights, runs 1000 steps with a fixed command, prints `track` and `linvel`, and renders three camera views.
```
python rollout.py 1 0 0 go1_final
```

| Command (x / y / yaw) | mean track | mean linvel (body frame) |
|---|---|---|
| 1 / 0 / 0 | **0.9405** | **0.961, 0.007, 0.018** |

For comparison, the standing-still policy had mean track 0.0187. The robot now walks forward at about 0.96 m/s. Videos are in `Tests/Test5` (leg motion score about 5 to 10, vs about 0.2 to 0.5 for standing). Full log: `Tests/Test5/Test5.md`.

### What changed vs the Colab runs

| | Colab (Tests 3, 4) | Test 5 (local) |
|---|---|---|
| `num_timesteps` setting | 18M / 25M | 100M |
| `num_evals` | 10 | 3 |
| **Steps actually run** | **about 29.5M** | **about 101.6M** |
| `num_envs` | 8192 | 4096 |
| GPU | T4 | RTX 5060 Laptop |
| Software | Colab's JAX, upgraded Flax | JAX 0.12.0, brax 0.14.2, flax 0.12.10 |
| Rollout env | notebook's eval env (yaw 2π) | default env (yaw ±1.2) |
| Command | x = 1, y = 0, yaw = 0 | the same |

Unchanged: task, reward scales, command config, PPO hyperparameters, network, domain randomization, seed.

The real step count (about 3.4x more) is the likeliest cause, but `num_envs`, hardware and software also changed, so it is not proven. A control run with the default `num_evals` (about 29.5M steps) on the laptop would test it.

### What `num_evals` does

It sets how many times training pauses to evaluate the policy: it splits training into `num_evals - 1` epochs, and after each one Brax runs evaluation episodes, calls `progress_fn` (the printed reward) and `policy_params_fn` (the checkpoint save). It does not change the learning algorithm. It does change the real total step count, because each epoch is rounded up. More evals give a finer reward curve and more checkpoints, at a small time cost.

### What matters in training (rough ranking, general RL knowledge; only the first is supported by my own runs)

1. **Tier 1, changes what can be learned:** total real steps; reward design; command setup; random seed (single runs can differ a lot).
2. **Tier 2, shapes learning:** learning rate, entropy cost, discount, batch size, unroll length, minibatches, updates per batch, `num_envs`, network size, domain randomization.
3. **Tier 3, bookkeeping and speed:** `num_evals` (but it changes the real step count through rounding), checkpoint saving, progress printing, warnings, GPU speed.
4. **Measuring, not training:** the rollout setup changes what I see and conclude, not what the policy learns.

Method: change one thing at a time and keep a table of what differed.

### Corrections to earlier entries
- The yaw 2π was not in training (training used ±1.2). The "wide yaw range" idea is dropped.
- "More steps hurt" (Test 4) is dropped. The 18M and 25M Colab runs both ran about 29.5M steps, and the late reward drop is probably run-to-run variation.
- The 100M run did not take 70 minutes. It took 19 minutes, about 92k steps/s, not 23k (the short test was dominated by compile and evaluation overhead).
- The undertrained explanation is now supported by the rollout, not only by the reward.

### ~03:40 — Mix-up: "18mil steps" videos that were really 100M

I rendered `go1_final` a second time, 20 minutes after the first, and put the clips in a folder named "18mil steps". `go1_final` was still the 100M weights (no 18M training had run), so they were the same policy twice. A frame comparison showed the two sets almost identical for the first about 10 s, then drifting apart (small GPU differences grow over time). Lesson: use one folder per run, and name videos by the run's folder and checkpoint.

### ~03:50 — Test 6: control run (about 29.5M steps, local)

**Setup:** `~/go1_ctrl`, `num_timesteps` 18M, `num_evals` 10, `num_envs` 4096, the same laptop, software and script as Test 5.

| | Test 6 (control) | Test 5 (walking run) |
|---|---|---|
| Steps run | **29,491,200** | 101,580,800 |
| Final reward | 19.613 | 26.248 |
| Training time | 7:24 (about 66k steps/s) | 18:23 (about 92k steps/s) |
| mean `track` (x = 1) | **0.0187** | **0.9405** |
| mean `linvel` | 0.002, -0.002, 0.002 | 0.961, 0.007, 0.018 |
| Videos | Stands still (motion about 0.2 to 0.3) | Walks (motion about 5 to 10) |

The step count matched the formula exactly (predicted 29,491,200). The videos are in `Tests/Test6/18mil steps/` (named after the 18M setting; actual steps 29,491,200). Log: `Tests/Test6/Test6.md`.

**Takeaway**
- Same machine, software, `num_envs` and script: about 29.5M steps stands still, and about 101.6M walks. Steps are the main driver in these runs.
- The Colab notebook was not broken. It reproduces the standing-still result.
- Reward around 19 to 20 went with standing, and 26 with walking (two policies, so a lead, not a rule).
- Caveat: one seed per run; `num_evals` also differed (10 vs 3), which should not affect learning.

### ~15:30 — Test 7: 120M run, diagonal commands

**Setup:** WSL folder `~/go1_test8` (the repo calls it Test 7), `num_timesteps` 120M, `num_evals` 16, `num_envs` 4096. `num_evals` 16 was chosen so the real total lands near 120M: it ran **122,880,000** steps (15 epochs of 8,192,000). Compile 0:59, training 29:31, about 69k steps/s.

**Reward:** final 26.936, peak 26.961 at 106.5M. It sat at about 19 from 24.6M to 57.3M steps (the standing level of the earlier tests), then climbed to about 27. I did not roll out the plateau checkpoints.

| Command (x / y / yaw) | mean `track` | mean `linvel` x, y | Distance from start | Yaw turned |
|---|---|---|---|---|
| 1 / 0 / 0 | 0.942 | 0.943, -0.003 | 18.8 m | -16.6° |
| 1 / 0.5 / 0 | 0.943 | 0.965, 0.464 | 21.2 m | +21.1° |
| 1 / -0.5 / 0 | 0.953 | 0.955, -0.477 | 21.2 m | -19.7° |
| 1 / -1 / 0 | 0.932 | 0.950, -0.930 | 26.0 m | -36.4° |

**Findings**
- All four commands are followed. That includes y = -1, outside the trained y range of ±0.8 (`command_config.a = [1.5, 0.8, 1.2]`). Out-of-range commands are not blocked, the policy just receives them as an input it never trained on.
- **The slow turn at yaw = 0.** The yaw command is a turn rate, not a heading. The mean yaw rate at 1 / -1 / 0 was -0.031 rad/s, which over 20 s is about -36° (the same number as the graph). The reward does not punish it (`tracking_ang_vel` is `exp(-error² / 0.25)`, about 0.996 at 0.03 rad/s), and the policy has no heading input (observation: body-frame velocity, gyro, gravity direction, joint angles and speeds, last action, command). So heading drift is expected, not a bug. The drift followed the sideways command in this run (+21°, -20°, -36°, -17° at y = 0.5, -0.5, -1, 0), but that is four points from one run.
- A heading-hold controller outside the policy (yaw command from the heading error) is the cheap fix, not retraining. Not built yet.
- The laptop's battery died during the rollouts. The training had already finished and saved, so only the rollouts were redone.

### ~17:10 — Test 7.5: Test 5 policy on a sideways command

**Question:** In Test 7 the Test 5 policy stood still on 1 / -1 / 0 while the 120M policy walked it. Is that about the range (y = -1 is out of ±0.8) or about sideways walking in general? Same `go1_final` (checked identical to `~/backup_run1`), only the command changes.

| Test 5 policy, command | mean `track` | mean `linvel` x, y | Distance from start |
|---|---|---|---|
| 1 / 0 / 0 | 0.941 | 0.963, 0.010 | 19.2 m |
| **1 / -0.5 / 0** | **0.935** | 0.958, -0.447 | 21.1 m |
| 1 / -1 / 0 | 0.0004 | 0.004, -0.003 | 0.09 m |

**Findings**
- The Test 5 policy walks y = -0.5, so it can walk sideways. Its failure at y = -1 is most likely the range. Not proven: only two y values, the combined speed could matter, and the two runs also differ in seed, steps and `num_evals`.
- The 120M policy extrapolated to y = -1 anyway: two policies, two outcomes on the same out-of-range command.

### Tools added (Scripts/)
- `rollout.py` now saves the motion data (`motion_<...>.npz`) and has `--no-video`. Without video it copies each step's numbers to the CPU right away, so nothing piles up on the GPU.
- `plot_motion.py`: distance, velocity against the command, and rotation (yaw angle (total) and yaw rate).
- `plot_trail.py`: a bird's-eye trail of the body centre and the head tip, both shaded from start to end. The head is 0.3 m ahead of the body centre in the model; `--head-length 2` is only a drawing aid for long walks.
- `plot_reward.py`: reward table and graph from a training log.
- I checked the Playground source: no chase or target-following task, and the gait modes (trot, walk, pace, bound, pronk) exist only for Spot and H1, not Go1. Only flat and rough Go1 terrains are wired up (stairs and bowl scene files exist but are unused).

### ~20:05 — Test 8: rough terrain (stopped early)

**Setup:** `~/go1_rough110`, `Go1JoystickRoughTerrain`, `num_timesteps` 110M, `num_evals` 18 (planned 111,411,200 steps, 17 epochs of 6,553,600), `num_envs` 4096. **The run stopped at 91,750,400 steps** (24 min, about 66k steps/s) with a GPU out-of-memory error (`Failed to allocate 524288 bytes on device 'cuda:0'`). No `go1_final` was saved, so the rollouts use `ckpt_91750400`. The cause is unknown.

**Reward:** 7.5 at 6.6M, 18.4 at 19.7M, about 19 to 20 from 20M to 66M, then a climb to 24.1 at 91.75M, still rising. Flat (Test 7) had 25.5 at about 90M.

| Command (x / y / yaw) | mean `track` | mean `linvel` x, y | Rollout length | Distance from start | Yaw turned |
|---|---|---|---|---|---|
| 1 / 0 / 0 | 0.929 | 0.965, 0.002 | 13.2 s | 12.6 m | +14° |
| 1 / -1 / 0 | 0.817 | 0.890, -0.832 | 7.6 s | 9.3 m | -19° |
| 1 / -1 / 1 | 0.914 | 0.930, -0.946 | 20 s | 0.24 m | +1077° (3 laps) |
| 0.5 / 0.5 / 3 | 0.848 | 0.604, 0.366 | 20 s | 0.05 m | +3208° (about 9 laps) |

**Findings**
- The policy follows all four commands on the bumpy ground, including a turn of 3 (the trained limit is 1.2). With a turn of 1 it walks a circle about 2.8 m wide; with a turn of 3, about 0.5 m wide.
- The rough patch is 20 m across (±10 m). A straight walk reaches the edge after 7 to 13 s, so those rollouts are stopped at 9.5 m from the centre and cover less than 20 s.
- The video rollouts for the two circling commands first crashed with GPU out-of-memory: a video rollout keeps every step's state on the GPU, and the rough states are bigger. Moving each state to the CPU fixed it, at about 11 minutes per video rollout.
- Videos: I looked at contact sheets (8 frames, 2.5 s apart) only. The robot stays upright, the legs change pose frame to frame, the ground is rocky and bumpy. Gait quality (sliding, hopping) is not checked.
- A quick check, not saved in the repo: the flat policy from Test 7 scored 0.46 on 1 / -1 / 1 on this terrain, against 0.914 here. Different training runs, so a hint only.
- Caveats: one seed, 82% of the planned steps, small bumps (up to about 5 cm). Logs: `Tests/Test7/Test7.md`, `Tests/Test7.5/Test7.5.md`, `Tests/Test8/Test8.md`.

### ~22:30 — How the Go1 model is built (read from the XML, counted with MuJoCo)

**Source of the files:** the five STL files come from the MuJoCo Menagerie repo (`deepmind/mujoco_menagerie`, `unitree_go1/assets/`, BSD-3 from Unitree). Playground does not ship them: on first load it runs `git clone` of Menagerie into `mujoco_playground/external_deps/mujoco_menagerie` and checks out a pinned commit (`1b86ece5…`, 2025-11-06). That is why the files in `go1env` are dated at my first run. The model file is `go1_mjx_feetonly.xml`.

**Counts (from `env.mj_model`)**

| Thing | Count |
|---|---|
| Mesh files | 5 (`trunk`, `hip`, `thigh`, `thigh_mirror`, `calf`) |
| Visible mesh pieces | 13 (trunk 1, hip 4, thigh 2 + thigh_mirror 2, calf 4) |
| Bodies (excluding the world) | 13: trunk + 4 legs × (hip, thigh, calf) |
| Joints | 13: one free joint (the trunk, 6 degrees of freedom) + 12 hinges |
| Actuators | 12 (one per hinge) |
| Degrees of freedom | 18 (6 body + 12 joint angles) |
| Hidden simple shapes (group 3) | 1 box, 12 cylinders, 20 capsules, 4 spheres |
| Shapes that can collide with the floor | **4 foot spheres only.** The box, cylinders and capsules have `contype` 0 and `conaffinity` 0 (checked), so they cannot collide with anything in this "feet only" model. The mesh pieces (group 2) have no collisions either. |
| Physics timestep / policy step | 0.004 s / 0.02 s |

**How it works**
- The pieces form a tree: trunk → hip → thigh → calf, each placed at a fixed offset from its parent. The 12 hinges say each piece can only rotate about one axis relative to its parent, within limits (for example, hip swing about ±0.86 rad). The trunk floats free in the world.
- The policy outputs 12 target joint angles. A spring-damper (PD) controller turns the gap between target and actual angle into a motor force, and the simulator integrates the forces and contacts.
- The simulator's whole state for the robot is 18 numbers (`qpos`: body position, orientation, 12 joint angles).
- Rendering copies those numbers into a MuJoCo `MjData`, runs `mj_forward` to place every body, and MuJoCo's OpenGL renderer draws the STL triangles at those places. The meshes are visual only (group 2, no collisions). The only live collision shapes are the four foot spheres. The hidden box, cylinders and capsules are defined in the file but switched off.
- A visual-only part (a hat) can be added without changing the physics: the trunk's mass is set explicitly, and a visual shape adds no joints, so `qpos` keeps the same shape. Tested: a hat drawn through `modify_scene_fns` in the render call (cylinders, one function per frame). Not tested: loading a custom STL (its bytes have to be in the assets dictionary under the filename used in the XML).

**Motors and settings (read from the XML, the config and the loaded model)**
- Each hinge has a position actuator with `kp` 35 and a torque limit of ±23.7 N·m, plus joint damping 0.5 (acts as Kd) and armature 0.005. The policy's action is a target angle (`action_scale` 0.5 around the default pose). So each joint behaves like a spring-damper: torque = Kp × (target − angle) − Kd × speed.
- The XML's default class says "This globally disables all collisions" (`contype` 0, `conaffinity` 0). Only the four foot spheres switch collisions back on.
- Mesh sizes (metres, no scale in the XML): the trunk mesh spans x -0.132 to 0.299, y ±0.097, z -0.061 to 0.073 (0.431 × 0.194 × 0.134 m, 112,384 triangles). The hip and calf meshes are about 30k triangles each. The trunk's `head` marker is at x = 0.3, at the front edge of the mesh. Trunk mass 5.2 kg, total 12.74 kg.
- Solver: Newton with 1 iteration per physics step. This is why `solver iterations limit reached` prints constantly. It is a speed trade-off, not an error. I have not measured whether it slows training.
- Timing: physics step 0.004 s, policy step 0.02 s (5 physics steps per policy step), 4096 robots in parallel on the GPU.

**How this compares with game engines** (general knowledge; I did not test a game engine)
- Same ingredients: rigid bodies, hinges, contacts, friction, springs. A ragdoll with hinges and elastic bands is the same idea as a joint with a spring-damper.
- Different goal: games aim for stable and good-looking; MuJoCo aims for accuracy about a specific robot (masses and joint limits from the real Go1) and for speed on many copies at once.
- MuJoCo stores the robot as joint coordinates (the 18 numbers), so hinges cannot pull apart. Many game ragdolls treat each piece as free and hold it with constraints that can stretch. This varies by engine.
- It is still approximate: one solver iteration, soft contacts. The gap to a real robot is why domain randomization is on in training.

**Where MuJoCo comes from:** first built at the University of Washington (2012 paper by Todorov, Erez and Tassa), sold by Roboti LLC, bought by Google DeepMind in October 2021, open-sourced under Apache 2.0 in May 2022. Sources: Wikipedia (MuJoCo) and OpenSource For You. I found no link between Fan Shi and MuJoCo itself; his co-authored paper (arXiv 2410.03076) is about differentiable simulation, and its abstract does not name the simulator.

### Oct 21 evening — Checkpoint sweep of the Test 7 run (command 1 / 0 / 0, flat, no video)

Each of the 16 checkpoints of the Test 7 run (`~/go1_test8`, every 8.2M steps) rolled out in `~/go1_sweep7`.

| Checkpoint (steps) | Reward (Test 7 log) | mean `track` | mean forward speed (m/s) | Behaviour |
|---|---|---|---|---|
| 0 | 0.001 | 0.018 | 0.004 | Stands |
| 8.2M | 8.3 | 0.019 | 0.003 | Stands |
| 16.4M | 16.9 | 0.019 | 0.002 | Stands |
| 24.6M | 19.3 | 0.019 | 0.003 | Stands |
| 32.8M | 19.0 | 0.019 | 0.002 | Stands |
| 41.0M | 19.3 | 0.019 | 0.002 | Stands |
| 49.2M | 19.2 | 0.019 | 0.002 | Stands |
| 57.3M | 20.1 | 0.019 | 0.002 | Stands |
| **65.5M** | 22.4 | **0.792** | 0.821 | **Walks** (slow, y drift -0.12) |
| 73.7M | 24.4 | 0.914 | 0.926 | Walks |
| 81.9M | 24.9 | 0.940 | 0.963 | Walks |
| 90.1M | 25.5 | 0.927 | 0.944 | Walks |
| 98.3M | 26.7 | 0.936 | 0.959 | Walks |
| 106.5M | 27.0 | 0.944 | 0.964 | Walks |
| 114.7M | 27.0 | 0.939 | 0.939 | Walks |
| 122.9M | 26.9 | 0.942 | 0.943 | Walks (final) |

**Findings**
- In this run walking appears **between 57.3M and 65.5M steps**, suddenly: `track` goes from 0.019 to 0.79 within one 8.2M-step gap, then creeps up to about 0.94. The exact step is not known (checkpoints are 8.2M apart).
- Reward misled before the switch: 16.9 at 16.4M and 20.1 at 57.3M while the robot did not move.
- `track` 0.019 is what standing still at x = 1 gives (`exp(-4)` is about 0.018). The untrained `ckpt_0` reads the same.
- Hypothesis, not tested: the tracking score is flat near zero when the robot is a full 1 m/s off, so there is little signal to follow until some version of the policy starts moving.
- Caveats: one seed, one command, one rollout per checkpoint.

### Oct 22 ~02:00 — Test 9: five more seeds (flat ground, no video)

**Setup:** seeds 1 to 5, `~/go1_seed1` to `~/go1_seed5`, same task and `num_envs` 4096, seed set with `seed=N` in `ppo.train` (every earlier run used the default seed 0). Seed 1: 60 evaluations (96,665,600 steps, a checkpoint every 1.64M). Seeds 2 to 5: 31 evaluations (98,304,000 steps, a checkpoint every 3.28M). Run as one overnight queue in tmux. Logs and reward graphs are in `Tests/Test9/`.

| Run | Evaluations | Steps | Training time | Final reward | Peak reward | Reward first reached 21 |
|---|---|---|---|---|---|---|
| seed 0 (Test 7) | 16 | 122.9M | 29:31 | 26.9 | 27.0 | 65.5M (20.1 at 57.3M) |
| **seed 1** | **60** | 96.7M | 21:51 | **0.2** | 18.7 | **never** |
| seed 2 | 31 | 98.3M | 20:18 | 26.1 | 27.3 | 52.4M |
| seed 3 | 31 | 98.3M | 20:13 | 26.9 | 26.9 | 55.7M |
| seed 4 | 31 | 98.3M | 19:47 | 26.9 | 27.3 | 55.7M |
| seed 5 | 31 | 98.3M | 20:15 | 26.4 | 26.7 | 59.0M |

**Findings**
- **Seeds 2 to 5 have the same reward shape as seed 0:** about 19 to 20 for a long stretch, a climb from about 50 to 59M steps, 26 to 27 at the end. "Reward 21" is a rough marker I chose (in seed 0 the robot stood at 20.1 and walked at 22.4). **They have not been rolled out, so walking is not confirmed.**
- **Seed 1 collapsed.** Reward rose to about 18.7 by 18M (the standing level), dropped to about 12 at about 34M, bumped to 16 at 75M, then fell to about 0.2 by 85M and stayed there. The sweep (16 checkpoints, 1 / 0 / 0) gave `track` 0.018 to 0.0195 and forward speed near zero at every checkpoint, so it never walked. A reward that low suggests very short episodes (such as falling over), but I have not seen a video.
- **Confounded:** seed 1 was the only run with 60 evaluations. It is not known whether the collapse comes from the seed or the schedule. Next: rerun seed 1 with 31 evaluations, and another seed with 60. (Done, see Test 9.1 below.)
- In the Colab Test 4 the reward also dropped late in training. It may have been the same kind of collapse (not known).
- **Timing:** training was faster than estimated (about 74k steps/s with 60 evaluations, about 81k steps/s with 31). Evaluation overhead is smaller than the 15 to 35 s per evaluation I had guessed.
- A seed is the starting number for all the randomness in training (initial weights, random commands, noise). Same seed and settings give nearly the same run (not bit-identical on a GPU).

### Oct 22 — The perceptive-quadruped paper (arXiv 2410.03076): how vision is used

Checked against the paper's PDF (`Reference/`, 8 pages, text extracted with pypdf). Authors: Luo, Song, Klemm, Shi, Scaramuzza, Hutter. Affiliations: Luo, Klemm and Hutter at ETH Zurich, Song and Scaramuzza at the University of Zurich, Shi at NUS.
- **Sensor:** a depth camera, not RGB. "The robot observes a 16x12 wide depth-camera input along with its velocity." The policy takes the flattened 16 by 12 depth input through a single dense layer, and the paper says a dedicated image pre-processing network may not be needed. An 84 by 84 depth map with a standard CNN is used in a separate comparison. The text speaks of obstacles "within its field of view" but does not say where the camera is mounted.
- **Simulation only:** the word "real" does not appear in the text, and no real-robot experiment is described ("hardware" appears only for hardware acceleration). The renderer is not named; the paper writes about "depth rays". The main method avoids the rendering gradient, and the text says poorly conditioned depth gradients near object edges are expected to hurt the learning signal.
- **Speed claims:** the point-mass navigation is solved "in seconds" (about 50,000 simulator steps), and the pixel-to-joint-angle navigation on the quadruped trains with similar sample efficiency to blind locomotion, "1.5e6 samples corresponds to 15 minutes of wall-clock time".
- **Tasks:** a planar point-mass robot for pixel-based navigation (it tracks a forward velocity while avoiding obstacles seen in the depth input), and a quadruped for locomotion plus navigation.
- **Quadruped and framework:** ANYmal C in MuJoCo MJX (the same framework as my Go1 runs), trained with SHAC (a differentiable-simulation method) and PPO. The wall-clock timings are on an Nvidia 3060 Ti.
- **Limitations stated:** the combined quadruped navigation converges to a local minimum with lower robustness. Noise robustness is tested only for the point-mass policy.
- **For my project:** my Go1 policy is blind (48 inputs, no terrain information). The paper's idea of a small depth or height input is the closest cheap step. MJX has ray casting (`mjx.ray`, checked in the installed version), and Playground's Go1 code does not use it.

### Oct 25 — Test 9.1: seed or schedule? (flat ground, no video)

**Setup:** the Test 9 settings with one change at a time: seed 1 again with 31 evaluations (`~/go1_seed1_e31`, 98.3M steps) and seeds 2 and 3 with 60 evaluations (`~/go1_seed2_e60`, `~/go1_seed3_e60`, 96.7M steps). Trained on Oct 22, about 21 to 22 minutes each. Rolled out on Oct 25: `go1_final` on 1 / 0 / 0, `--no-video`, one 20 s episode. Full write-up in `Tests/Test9.1/Test9.1.md`.

| Run | Evaluations | Final reward | `track` | Forward speed | Result |
|---|---|---|---|---|---|
| seed 1 | 31 | 27.06 | 0.968 | 0.953 m/s | Walks |
| seed 2 | 60 | 20.05 | 0.0187 | 0.001 m/s | Stands still |
| seed 3 | 60 | 12.78 | 0.0187 | 0.001 m/s | Stands still |

**Findings**
- **Seed 1 is not a broken seed.** The same seed trains to 27.06 with 31 evaluations and walks. With 60 evaluations it collapsed (Test 9).
- **All three 60-evaluation runs failed to walk** (seed 1 collapsed to 0.17, seeds 2 and 3 stuck below 21 and stand still). The walking runs used 16 or 31 evaluations. In my runs, this points at the 60-evaluation schedule and not the seed.
- **`num_evals` is a training parameter here, not only a plotting setting.** It sets how often the reward is measured and a checkpoint is saved, but changing it changed the outcome for the same seed. It also changes the steps actually run (96.7M at 60 against 98.3M at 31), the order of the random numbers, and, by my reading of the Brax code, the reset schedule (all environments reset every 40 steps at 60 evaluations, 80 at 31, 200 at 16). Which of these matters is not tested. Test: 60 evaluations with `num_resets_per_eval` 5. Until then, keep `num_evals` the same for runs that are compared.
- **Standing, not fallen:** the saved motion data shows the 60-evaluation robots at about 0.3 m body height and upright at the end, moving 2 to 3 cm in 20 s. No video was checked.
- **Reward misleads again:** seed 3 ended at 12.78 and seed 2 at 20.05, yet both give `track` 0.0187.
- **Caveats:** one rollout per policy, one command, 3 runs at 60 against 6 at 16 or 31. Intermediate values such as 45 are not tried.

### Oct 26 — Test 9.2: is it the reset interval? (flat ground, no video)

**Setup:** the Test 9.1 settings with `num_resets_per_eval` set directly, so the reset interval (every 40 or 80 steps) is chosen and not left to `num_evals`. Seeds 2 and 3 at 60 evaluations with 5 resets (interval 80) and at 31 evaluations with 20 resets (interval 40), plus seed 4 at 60 evaluations with the default (interval 40). Seed 4 with 5 resets stopped at 52.4M steps when the laptop was shut off. About 21 minutes per run. Rolled out with `Scripts/sweep.py` (`go1_final` of seed 1 gave 0.9676 against 0.9678 from `rollout.py`). Full write-up in `Tests/Test9.2/Test9.2.md`.

| Run | Evaluations | Reset every | Final reward | `track` | Result |
|---|---|---|---|---|---|
| seed 2 | 60 | 80 | 26.53 | 0.955 | Walks |
| seed 2 | 31 | 40 | 18.63 | 0.018 | Stands still |
| seed 3 | 60 | 80 | 26.61 | 0.971 | Walks |
| seed 3 | 31 | 40 | 18.53 | 0.019 | Stands still |
| seed 4 | 60 | 40 | 14.80 | 0.018 | Stands still |
| seed 4 (rerun, Nov 2) | 60 | 80 | 27.18 | 0.957 | Walks |
| seed 2 (Oct 31, 4 resets, about 116M steps) | 60 | 120 | 27.07 | 0.959 | Walks |

**Findings**
- **The reset interval decides it, not `num_evals`.** Changing only `num_resets_per_eval` turned a failing setup into a walking one and a walking setup into a failing one, for two seeds. Every 80 steps walks, every 40 steps stands, whichever `num_evals` is used.
- **Each seed keeps its own switch time.** Seed 2 switches at about 50M steps in both of its walking runs, seed 3 late (after 68M) in both of its.
- **All five runs at 31 evaluations walk at the end** (`track` 0.952 to 0.968). The switch is sudden, between about 49M and 65M steps for four of them. Seed 3's reward passed 21 at 55.7M while it still stood still, so reward misled me again.
- **Caveats:** the Brax code reading is mine and not checked line by line. Only 40, 80 and 120 were tried, and only multiples of 40 can be set, so I cannot place the line between 40 and 80. One rollout per checkpoint, one command. Seed 4 was tested once per setting (stands at 40, walks at 80 in the rerun), so it agrees but is not a full pair. I do not know why short intervals stop the walking.
- **Later runs (Oct 31 to Nov 2):** a reset every 120 steps also walks (seed 2, `track` 0.959, about 116M steps, switch between 60.9M and 64.9M). The seed 4 rerun walks at 80 (switch between 60.6M and 63.9M). The `track` spike of 0.032 at 52.4M in one seed 3 run was a fall: that checkpoint tipped over and went through the floor, because only the four foot spheres collide in this model.
- **Process lesson:** my first overnight run was one 3.8-hour chain with the plots and the copy at the end. The laptop was shut off and nothing from the night was in the repo. Now one unit (a run and its sweep) at a time, copied and plotted before the next.

### Oct 27 — Test 9.3: the Test 7 policy on my new sine ground (no training)

**Setup:** `go1_final` of the Test 7 run (flat ground, seed 0, 122.9M steps), command 1 / 0 / 0, one 20 s episode with `PRNGKey(0)` on three grounds, using `Scripts/rollout_terrain.py`. The new ground is `Assets/Grounds/SimpleUniFormRoughGround_1.xml` (100 x 100 m, two crossing sine waves, 0.4 m wavelength, crests 1x to 1.5x; written by Claude from my design). Full write-up in `Tests/Test9.3/Test9.3.md`.

| Ground | `track` | Distance in 20 s | Fell |
|---|---|---|---|
| New sine ground | 0.9205 (rerun 0.9199) | 18.58 m | no |
| New sine ground with the cone hat | 0.9203 | 18.57 m | no |
| Stock flat | 0.9404 | 19.20 m | no |
| Stock rough (20 m patch) | 0.887 over the first 612 steps, 0.543 over the whole episode | 17.70 m | yes, at 12.2 s |

**Findings**
- **The ground loads and simulates in Playground** (Warp backend), and a policy trained on flat ground only walks across it with `track` close to its flat value. The robot stayed upright (never below 0.988).
- **The stock rough "fall" is the edge of the patch.** The patch is 20 m wide, the robot crosses 10 m from the centre at about 11.8 s and is terminated at 12.2 s. Before that, `track` was 0.887.
- **The cone hat made no visible difference** (0.9203 against 0.9199), but with one run each I cannot tell "no effect" from a small one. It loads in Playground as a visual-only geom.
- **Videos and plots** exist for both runs (`Tests/Test9.3/x1.0_y0.0_yaw0.0/`). Rendering this ground takes about 17 minutes per camera (3 parallel renders: about 20 minutes in total).
- **Caveats:** one path of about 19 m in a 100 m patch, one policy, one command, no training. Nothing here says the ground is trainable or how much GPU memory it needs. Videos spot-checked (one frame each), not watched in full.

### Nov 4 — Test 10: the Test 7 policy on a ramp ground and a stair ground (no training)

**Setup:** `go1_final` of the Test 7 run (flat ground only), `go1_mjx_feetonly.xml` (only the feet collide), command **0.85 / 0 / 0** held for 65 s (3,250 steps), same start (`PRNGKey(0)`) on both grounds, `Scripts/rollout_terrain.py`. Grounds from `Scripts/make_terrain.py` (written by Claude from my layout): 100 x 100 m square rings, flat to d = 10 m, a 15° ring to 18 m, a plateau to 28 m, a 25° ring to 35 m, a plateau to 45 m, a 35° ring to 50 m. The stairs have the same average angles (0.40 m tread and 0.107 m rise on the 15° ring). Full write-up in `Tests/Test10/Test10.md`.

**Step 1, climb search** (ramp ground, 80 s each; arrived = d >= 18.5 m, z >= 2 m above the start, upright):

| Command (m/s) | 0.30 to 0.75 | 0.80 | **0.85** | 0.90 | 1.10 | 1.30 | 1.50 |
|---|---|---|---|---|---|---|---|
| Result | stuck at the foot | still climbing at 80 s | **arrived at 61.3 s** | 44.9 s | 27.6 s | 23.3 s, fell at 78 s | 21.4 s, fell at 31 s |

No speed got past the 25° ring.

**Step 2, the two 65 s runs**

| | Ramp (`slanted`) | Stairs (`staired`) |
|---|---|---|
| Fell | no | **yes, at step 758 (15.2 s)** |
| Mean `track` | 0.483 | 0.259 |
| Distance from start | 26.64 m | 11.33 m |
| First plateau (d = 18 m) | reached at 60.5 s | never |
| Upright | min 0.970 | min -0.980, on its back from 15 s |

**Findings**
- **Ramp:** about 0.8 m/s on the flat middle, about 0.4 m/s while moving on the 15° ramp, with stops of 1 to 3 s (25.1 to 26.3 s, 38.0 to 41.4 s, 48.2 to 51.4 s). It crossed the 8 m of ramp in about 46 s, then went back to 0.8 m/s on the plateau. Heading stayed within 6°.
- **Stairs:** it reached the first riser at 14.7 s at full speed, spiked to 1.5 m/s while turning about -235° in about 1 s, and was terminated at 15.2 s. After that nothing moved for 50 s (speed 0.00, body height -0.23 m, upright -0.98).
- **Same average angle, opposite outcome,** so the angle is not what stops it. The policy has no height input and never saw a slope or a step.
- **The track videos agree with the numbers.** Ramp: a steady trot, ramp edge visible at about 12 s. Stairs: normal trot to about 12 s, then from 16 s on only four foot spheres against the step surface, unchanged to the end. My reading, not tested: with feet-only collision, the trunk has nothing to rest on once it is upside down and sinks under the terrain.
- **Caveats:** one policy, one seed, one start, one run per ground. 0.85 m/s was chosen for the ramp, not for the stairs. The track videos were checked by one frame every 4 s; `top` and `side` were not watched.

### Nov 7 — Test 10.1: the ramp ground with command 2 / 2 / 0 (no training)

**Setup:** as Test 10 (Test 7 `go1_final`, feet-only robot, `SlopeRingsGround_1`, 65 s, `PRNGKey(0)`, `Scripts/rollout_terrain.py`), but the command is **2 / 2 / 0**, above the trained range (`command_config.a = [1.5, 0.8, 1.2]`). Script `Tests/Test10.1/run_test10_1.sh`, write-up `Tests/Test10.1/Test10.1.md`.

| | Test 10 slanted (0.85 / 0 / 0) | Test 10.1 (2 / 2 / 0) |
|---|---|---|
| Fell | no | **yes, at step 69 (1.4 s)** |
| Mean `track` | 0.483 | 0.0000 |
| Distance from start | 26.64 m | 0.61 m |
| Upright | min 0.970 | min -0.996, on its back from 1.4 s |

**Findings**
- **It never walks.** For the first second body speed stays within about -0.17 to +0.16 m/s (command 2 / 2), then it tips sideways (upright 0.93 at 1.0 s, 0.81 at 1.2 s, -0.12 at 1.38 s; sideways speed spike about 1.7 m/s during the roll). Terminated at 1.4 s, on the flat middle, so the ramp played no part.
- **No recovery:** nothing moves from 1.6 s to 65 s (upright -0.98, body about 0.25 m below the start level). The track video shows only foot spheres for the rest of the run, as on the Test 10 stairs.
- **Out-of-range commands are not just ignored.** In Test 7 y = -1 (0.2 over the limit) was followed, and here x 2 (0.5 over) and y 2 (1.2 over) together fall within 1.4 s. I cannot tell which input does it, since both were out of range at once.
- **Caveats:** one run, one start. Track video checked by frames, `top` and `side` not watched.

### Nov 7 — Test 10.2: the ramp ground with command 1.5 / 0.8 / 0, the trained maximum (no training)

**Setup:** as Test 10 and 10.1 (Test 7 `go1_final`, feet-only robot, `SlopeRingsGround_1`, 65 s, `PRNGKey(0)`, `Scripts/rollout_terrain.py`), command **1.5 / 0.8 / 0**, the edge of the trained range. I checked the limit in the installed Playground code (`go1/joystick.py`: `command_config.a = [1.5, 0.8, 1.2]`, commands drawn uniformly from -a to +a, no clip). Script `Tests/Test10.2/run_test10_2.sh`, write-up `Tests/Test10.2/Test10.2.md`.

| | Test 10 (0.85 / 0 / 0) | Test 10.1 (2 / 2 / 0) | Test 10.2 (1.5 / 0.8 / 0) |
|---|---|---|---|
| Fell | no | yes, 1.4 s | **yes, step 1340 (26.8 s)** |
| Mean `track` | 0.483 | 0.0000 | 0.243 (0.607 over the first 25 s) |
| Distance from start | 26.64 m | 0.61 m | 29.97 m |
| 15° ramp reached (d = 10 m) | 14.7 s | never | 7.2 s |
| First plateau (d = 18 m) | 60.5 s | never | 16.3 s |
| Furthest ring distance | 20.7 m | 1 m | 28.08 m |

**Findings**
- **It follows the command at the trained maximum:** 1.49 m/s forward and 0.75 m/s sideways on the flat middle, so 2 / 2 / 0 failed because it was out of range, not because it was fast (but x and y were both out of range, so I still cannot say which).
- **Much easier ramp climb:** 15° ramp crossed in about 9 s (7.2 to 16.3 s) at 0.84 / 0.53 m/s, without stopping, against about 46 s at 0.85 / 0 / 0. Full command again on the plateau (1.51 / 0.75 m/s).
- **The 25° ramp stops it:** reached d = 28.0 m at 25.5 s, forward speed to about 0 by 26.4 s, tumbled at 26.8 s (sideways speed spike about 1.8 m/s), then lay on its back (upright -0.995, body height 1.86 m, below the plateau surface) until 65 s. Same fall mode as Test 10 and 10.1.
- **Heading drift:** yaw drifts to about +85° in 25 s with command yaw 0, so the path curves.
- **Caveats:** one run, one start. Track video checked by frames, `top` and `side` not watched. The fall position is the body centre, not the feet.

### Nov 8 — Test 10.3: trained on the ramp ground, then Test 10.2's rollout again

**Training:** `Scripts/train_terrain.py` (written by Claude), run by `Tests/Test10.3/train_slanted.sh` in `~/go1_slanted`. Ground `SlopeRingsGround_1.xml` loaded under the task name `rough_custom`, feet-only robot, PPO settings of `Go1JoystickRoughTerrain`, stock domain randomization, `num_timesteps` 120M, `num_evals` 16, `num_envs` 4096, seed 0 (the Test 7 settings). Smoke run first (16.4M steps, 5:06, 4,540 MiB). **Full run: 122,880,000 steps, jit 1:03, training 24:44, peak 4,574 MiB, no crash.** The floor is geom 0 as the randomizer assumes (the script checks). Log `Tests/Test10.3/train.log`, plot `reward_plot.png`.

| Steps | 24.6M | 49.2M | 65.5M | 73.7M | 98.3M | 122.9M |
|---|---|---|---|---|---|---|
| Test 7, flat | 19.28 | 19.19 | 22.41 | 24.41 | 26.70 | 26.94 |
| Test 10.3, ramp ground | 19.49 | 19.31 | 20.57 | 22.21 | 24.29 | 25.74 |

**Rollout:** as Test 10.2 (command 1.5 / 0.8 / 0, 65 s, ramp ground, `PRNGKey(0)`), `Tests/Test10.3/run_test10_3.sh`, write-up `Tests/Test10.3/Test10.3.md`.

| | Test 10.2 (flat-trained) | Test 10.3 (ramp-trained) |
|---|---|---|
| Fell | yes, step 1340 (26.8 s) | yes, step 1156 (23.1 s) |
| Mean `track` | 0.243 | 0.197 |
| Distance from start | 29.97 m | 28.02 m |
| Crossed the 15° ramp (d = 18 m) | 16.3 s | 15.7 s |
| Body speed on the 15° ramp (fwd / side) | 0.85 / 0.54 m/s | 0.86 / 0.47 m/s |
| Reached the 25° ramp (d = 28 m) | 25.5 s | 22.0 s |
| Slowest forward speed | -0.18 m/s | -1.68 m/s |

**Findings**
- **No gain from training on this ground, as it is.** Same speeds on the flat, the 15° ramp and the plateau within about 0.1 m/s, and the 25° ramp still ends in a fall, 3.7 s earlier. One seed, so I do not read the difference as worse.
- **A different fall:** it climbed about 0.4 m up the 25° ramp (body height 2.64 m against a standing 2.46 m on the plateau), slid backwards (forward speed -1.68 m/s) and flipped; the flat-trained one stalled at the foot and tumbled sideways. Both on their back (upright -0.98) until 65 s.
- **My guess, untested:** every training episode starts within 0.5 m of the centre and lasts 20 s, so the 25° ring (28 m out) may almost never be reached. Not measured.
- **Caveats:** one training run, one rollout, one command. Training rewards are on different grounds. Track video checked by frames, `top` and `side` not watched.

### Nov 8 — Test 10.4: the ramp-trained policy with command 1.5 / 0 / 0

**Setup:** as Test 10.3 (policy `~/go1_slanted/go1_final`, `SlopeRingsGround_1`, 65 s, `PRNGKey(0)`), command **1.5 / 0 / 0** instead of 1.5 / 0.8 / 0, to see if the sideways part was the problem. Script `Tests/Test10.4/run_test10_4.sh`, write-up `Tests/Test10.4/Test10.4.md`.

| | Test 10.2 (flat-trained, 1.5 / 0.8) | Test 10.3 (ramp-trained, 1.5 / 0.8) | Test 10.4 (ramp-trained, 1.5 / 0) |
|---|---|---|---|
| Fell | step 1340 (26.8 s) | step 1156 (23.1 s) | step 1316 (26.3 s) |
| Mean `track` | 0.243 | 0.197 | 0.239 |
| Time on the 15° ramp (d 10 to 18 m) | 9.1 s | 8.5 s | 10.0 s |
| Forward speed on flat / 15° ramp / plateau | 1.49 / 0.85 / 1.51 | 1.45 / 0.86 / 1.46 | 1.44 / 0.86 / 1.44 m/s |
| Reached the 25° ramp (d = 28 m) | 25.5 s | 22.0 s | 25.0 s |
| Highest body height | 2.52 m | 2.64 m | 2.53 m |

**Findings**
- **The sideways command is not what stops the 25° ramp.** With y = 0 the policy stalls at the foot (25.0 s), crouches (forward speed 1.43 at 24 s, 0.95 at 25 s, -0.35 at 26 s) and topples at 26.3 s, then lies on its back (upright -0.987) until 65 s. Test 10.3's earlier fall (23.1 s, backward slide) looks like the odd one out.
- **Forward speed on the 15° ramp is about 0.85 m/s in all three 1.5-command runs**, flat-trained or ramp-trained, with or without the sideways command. In Test 10 the flat-trained policy at forward command 0.85 managed about 0.4 m/s moving on the same ramp, so the ramp seems to take roughly half the forward command (four runs, a pattern and not a measurement).
- **The ring-distance time on the ramp is longer at y = 0** (10.0 s against 8.5 s) only because the sideways speed (0.47 m/s) added to the speed along the ring normal in 10.3.
- **Caveats:** one run, one start. Track video checked by frames, `top` and `side` not watched.

### Nov 8 — Test 10.5: the ramp-ground policy trained for 253,952,000 steps (in three parts), then the Test 10.4 rollout

**Training:** `Tests/Test10.5/train_250M.sh` (250M requested, 32 evaluations, 4096 envs, seed 0, reset interval 200 steps as in Test 7 and 10.3). **Crashed after 122,880,000 steps** with `Warp CUDA error 2: out of memory (wp_alloc_device_async)` (GPU log flat at 4,578 of 8,151 MiB). Part 2 (`train_250M_part2.sh`, `--resume` from `ckpt_122880000`, seed 1) ran 98,304,000 steps and crashed the same way. Part 3 (`train_250M_part3.sh`, from `ckpt_98304000`, seed 2) ran 32,768,000 steps and finished. Total **253,952,000**. Only the weights are restored (optimizer state and step counter restart); the first evaluation of each resumed part (26.43, 26.93) confirms the weights came back. The crash is not at a fixed epoch (15, 12 and, in Test 8, 14) but after about 21 to 28 minutes of a process; cause unknown.

| Steps | Eval reward, mean (min to max) |
|---|---|
| 106M to 123M | 26.3 (26.0 to 26.6) |
| 123M to 180M | 26.7 (26.3 to 27.3) |
| 180M to 254M | 26.7 (26.0 to 27.8), last 26.96 |

**Rollout:** as Test 10.4 (ramp ground, command 1.5 / 0 / 0, 65 s, `PRNGKey(0)`), policy `~/go1_slanted_250m_part3/go1_final`. Write-up `Tests/Test10.5/Test10.5.md`, plot `reward_plot.png` (`Scripts/plot_reward_parts.py`).

| | Test 10.4 (122.88M steps) | Test 10.5 (253.95M steps) |
|---|---|---|
| Fell | step 1316 (26.3 s) | step 1655 (33.1 s) |
| Mean `track` (until the fall) | 0.239 (0.591) | 0.242 (0.475) |
| Time on the 15° ramp | 10.0 s | 17.1 s |
| Forward speed on flat / 15° ramp / plateau | 1.44 / 0.86 / 1.44 m/s | 1.47 / 0.59 / 1.47 m/s |
| Reached the 25° ramp | 25.0 s | 32.4 s |
| Highest body height | 2.53 m | 2.52 m |

**Findings**
- **No gain from doubling the steps.** The training reward is flat after about 110M steps (same as the flat run: 26.7 at 98M, 26.9 at 123M), and the policy stalls at the foot of the 25° ramp and topples 0.7 s after reaching it, as in Test 10.4. The fall is later only because the 15° ramp took 7.4 s longer.
- **Slower on the 15° ramp** (0.59 against 0.86 m/s forward, sideways drift of 0.3 to 0.5 m/s at 9 to 14 s, about -30° heading change). One run, with resets of the optimizer and changed seeds at the joins, so I cannot say whether it is an effect of the steps.
- **After falling it keeps moving its legs** (joint-angle spread 0.086 rad against 0.012 and 0.009 in 10.3 and 10.4; body rotation rate 0.38 against 0.04 and 0.05 rad/s) but stays on its back (upright -0.94 to -1.0).
- **Caveats:** one policy, one rollout, a resumed training in three parts. Track video checked by frames, `top` and `side` not watched.

### Insights so far
1. Judge walking by rollout numbers and video, not by reward.
2. Plan at least about 100M steps (the official setup is 200M). Standing is a stable local optimum that takes many steps to leave.
3. Always read the real step count from `last step count`, not from `num_timesteps`.
4. One folder per run, so results do not overwrite each other.
5. A command outside the trained range is not blocked. Behaviour varies: the 120M policy followed y = -1, the Test 5 policy did not (two runs, not a rule).
6. Sideways and diagonal walking works for commands inside the range (both policies).
7. Heading drifts slowly because yaw is a rate and the policy has no heading input. The fix belongs outside the policy.
8. Video rollouts hold every state on the GPU, so long ones can run out of memory. Move the states to the CPU, or skip the video.
9. On rough ground the policy followed every command tried (one run, stopped at 82%).
10. Looks and physics are separate: 13 mesh pieces from 5 STL files for the picture, 4 foot spheres as the only live collision shapes, 12 hinges as the motors.
11. In the one run checked with rollouts, walking starts suddenly between 57.3M and 65.5M steps. The reward plateau at 19 to 20 is standing. About 80M to 100M steps is a sensible budget, and more barely helps (26.7 at 98M vs 26.9 at 123M).
12. Training can collapse (seed 1: reward fell to 0.2), so one seed is not enough. Test 9.1 points at the 60-evaluation schedule and not the seed (seed 1 walks at 31 evaluations, seeds 2 and 3 at 60 stand still), in my runs.
13. `num_evals` changes the training, not only the plot, so keep it the same for runs that are compared. A reward of 12.8 and one of 20.0 can both be a robot that stands still.
14. The vision in the lab's paper is a tiny simulated depth image (16 by 12), fed to a single dense layer, with no real-robot experiment described (checked in the PDF text).
15. The reset interval is what separates walking from standing here: every 80 steps or more walks (80, 120, 200), every 40 stands, whatever `num_evals` is (Test 9.2, two seeds in both directions, a third seed agrees). Minimum for walking: that interval plus about 50M to 70M training steps.
16. Run long jobs in small units, each copied to the repo and plotted before the next starts.
17. A policy trained on flat ground walks on my new sine ground (`track` 0.92 against 0.94). The stock rough terrain is a 20 m patch, so a 20 s walk at 1 m/s leaves it. Whether the new ground can be trained on is not tested.
18. The flat-trained policy climbs a 15° ramp at 0.85 m/s or faster, slowly and in stop-and-go, but falls at the first 15° stair and does not get up (Test 10, one run each). Same average angle, so the steps matter, not the angle.
19. Raising the command far above the trained range does not give more speed: 2 / 2 / 0 made the policy fall within 1.4 s on flat ground (Test 10.1, one run, both inputs out of range at once).
20. At the trained maximum (1.5 / 0.8 / 0) the policy follows the command and crosses a 15° ramp in about 9 s, against about 46 s at 0.85 / 0 / 0. The 25° ramp still makes it fall (Test 10.2, one run).
21. A policy trained on the ramp ground itself (120M steps, 26 min, 4.5 GB, reward 25.7) did not do better on the 25° ramp than the flat-trained one at 1.5 / 0.8 / 0 (Test 10.3, one seed). Every episode starts in the flat middle, so the 25° ring may rarely be seen in training (untested).
22. Removing the sideways command (1.5 / 0 / 0, ramp-trained policy) does not change the 25° result: it stalls at the foot at 25.0 s and falls at 26.3 s (Test 10.4, one run). Forward speed on the 15° ramp was about 0.85 m/s in all three 1.5-command runs. The 25° ramp has beaten every policy and command tried so far.
23. Doubling the training steps on the ramp ground (254M, resumed in three parts) did not help: the reward stays near 26.7 after about 110M steps and the policy still stalls at the foot of the 25° ramp (Test 10.5, one run). Long runs crash with a CUDA out-of-memory error after 21 to 28 minutes of a process (Test 8, 10.5 twice), so restart them in parts of 10 to 12 epochs; cause unknown.

### Still unknown
- Whether x = 2 or y = 2 causes the Test 10.1 fall, and where between 1.5 / 0.8 (walks) and 2 / 2 (falls) the policy stops following.
- Whether the faster 15° ramp climb at 1.5 / 0.8 comes from the x part or the y part, and whether any command gets it up the 25° ramp.
- What causes the out-of-memory crash of long training runs (GPU log flat at 4.5 GB of 8 GB), and whether the slower 15° ramp climb and the leg waving after the fall in Test 10.5 are real (one run, three resumed parts).
- How often training episodes on the ramp ground reach the rings, and whether random start points or a slope curriculum would let a policy learn the 25° ramp.
- Why it falls on the stairs (riser, step height or speed), whether another speed gets up them, and whether the trunk sinking after a fall is the feet-only collision file (Test 10, none tested).
- Whether the new ground can be trained on (time, GPU memory, whether it learns to walk), and whether the hat has a small effect hidden in the run-to-run spread.
- Why short reset intervals stop the walking (the line is between 40 and 80 steps and cannot be set more finely; my guess is that 40 steps does not leave time to get up to speed, untested).
- Whether seeds 2 to 5 walk, and whether the switch lands in the same window for each (needs rollouts).
- Why the 60-evaluation schedule stops the walking (the reset-chunk idea is untested), where between 31 and 60 the line is, and what the collapsed robot does (needs a video).
- Whether `num_envs` 8192 changes the result.
- Whether the policy walks backward and turns in place (diagonals work).
- How good the gait is (speed variation, sliding, hopping).
- How much results vary between seeds, and whether the drift follows the sideways command.
- Whether the full 111M steps on rough ground would change the result, and why the GPU ran out of memory at epoch 14.
- Whether the constant solver warnings slow training.

### Next
See `todo.md` for the running list. Current first items: measure how often training episodes on the ramp ground reach the 15° and 25° rings, then (if the 25° ring is rarely seen) train with random start points or a growing slope; split the sideways and forward parts of the 1.5 / 0.8 and 2 / 2 commands; and a plateau-start sweep on the 25° ramp.
