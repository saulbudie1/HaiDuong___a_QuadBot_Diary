# Test Log 9.2 — Is it the reset interval? (flat ground, local RTX 5060)

**Date:** 2026-10-26 (queue started 04:18, laptop shut off before it finished); run G on 2026-10-31; plots, copy into the repo and videos on 2026-11-02
**Result:** In all four pairs the reset interval decided the outcome. Runs that reset every 80 steps **walk** (`track` 0.955 and 0.971) and runs that reset every 40 steps **stand still** (`track` 0.018 to 0.019), whatever `num_evals` is. A later run with a reset every 120 steps (G) also **walks** (`track` 0.959). Five of the six planned runs finished. The sixth (seed 4, 60 evaluations, `num_resets_per_eval` 5) stopped at 52.4M steps when the laptop was shut off and is still not rerun.

**Purpose:** [Test 9.1](../Test9.1/Test9.1.md) showed that `num_evals` changes the outcome: seeds 1, 2 and 3 with 60 evaluations did not walk, while the same seeds at 31 walked (seed 1) or had walking curves (seeds 2 to 5 of [Test 9](../Test9/Test9.md)). My reading of Brax `train.py` (not checked line by line) is that each evaluation interval is split into `num_resets_per_eval` chunks and all environments reset at the start of each chunk, so the reset interval is 40 steps times k, where k = ceil(`num_timesteps` / ((`num_evals` - 1) × 163,840 × `num_resets_per_eval`)). At 96M steps that gives 40 steps at 60 evaluations, 80 at 31 and 160 at 16 (Test 7 asked for 120M, which gives 200). Only multiples of 40 are possible. (This line first had 1,638,400 in place of 163,840 × `num_resets_per_eval`, which only agrees at the default of 10 resets; corrected on 2026-11-02 after planning run G.) This test changes `num_resets_per_eval` to set the interval directly, so the evaluation count and the interval are separated:
- 60 evaluations with `num_resets_per_eval` 5: interval 80 (like the walking e31 runs).
- 31 evaluations with `num_resets_per_eval` 20: interval 40 (like the failing e60 runs).

## Setup

Same as Test 9.1 (`train_test.py` copied into a new folder per run, step count, evaluations, resets and seed set by `sed`, `num_envs` 4096, 96,000,000 requested steps). The queue script is `run_queue.sh` in this folder.

| Run | seed | `num_evals` | `num_resets_per_eval` | Reset every | Steps actually run | Time (JIT + training) | Folder in WSL |
|---|---|---|---|---|---|---|---|
| A | 2 | 60 | **5** | **80** | 96,665,600 | 0:01:01 + 0:20:51 | `~/go1_t92_s2_e60_r5` |
| B | 2 | 31 | **20** | **40** | 98,304,000 | 0:01:00 + 0:20:35 | `~/go1_t92_s2_e31_r20` |
| C | 3 | 60 | **5** | **80** | 96,665,600 | 0:00:59 + 0:20:50 | `~/go1_t92_s3_e60_r5` |
| D | 3 | 31 | **20** | **40** | 98,304,000 | 0:00:58 + 0:20:18 | `~/go1_t92_s3_e31_r20` |
| E | 4 | 60 | 10 (default) | 40 | 96,665,600 | 0:00:57 + 0:21:21 | `~/go1_t92_s4_e60` |
| F | 4 | 60 | 5 | 80 | **stopped at 52,428,800** | none | `~/go1_t92_s4_e60_r5` (33 checkpoints, no `go1_final`) |
| G | 2 | 60 | **4** | **120** | 115,998,720 | 0:00:57 + 0:23:35 | `~/go1_t92_s2_e60_r4` |

- The step counts match the runs they are compared with, because the number of steps per epoch comes out the same (3.28M at 31 evaluations, 1.64M at 60). Run G is the exception: 60 evaluations with 4 resets gives 1,966,080 steps per epoch, so it ran about 116M steps, which is what the formula predicted (115,998,720).
- Run G was added after the queue (started 2026-10-31 14:17 with [run_r4.sh](./run_r4.sh), seed 2 so it can be compared with A and B). Its sweep used every second checkpoint (about 3.9M apart), so its onset window is wider than the others.
- Raw logs were filtered while they were written (solver and linesearch warnings removed), so they are small.
- The queue also swept (rolled out) the 40M to 70M checkpoints of the Test 9 and Test 9.1 runs. All five runs at 31 evaluations (Test 9 seeds 2 to 5 and the Test 9.1 seed 1) walk at the end (`track` 0.952 to 0.968). The onset windows are in the todo and will go into Test9.md.

## Rollout numbers

Command 1 / 0 / 0, one 20 s episode per checkpoint, no video, with `Scripts/sweep.py` (the policy is compiled once, the episode is the same as `rollout.py`). `go1_final` of seed 1 gave `track` 0.9676 here against 0.9678 from `rollout.py`, so the numbers match to about 0.0002. `track` about 0.019 means standing still, about 0.95 means walking.

| Pair | Run | Reset every | Final reward | `track` at the end | Swept checkpoints, 40M to 70M |
|---|---|---|---|---|---|
| seed 2 | e31 (Test 9) | 80 | 26.11 | 0.958 | stands up to 49.2M, walks from 52.4M |
| seed 2 | e60 (Test 9.1) | 40 | 20.05 | 0.019 | stands at every checkpoint |
| seed 2 | **A: e60, resets 5** | **80** | **26.53** (peak 27.58) | **0.955** | stands at 47.5M, walks from 50.8M |
| seed 2 | **B: e31, resets 20** | **40** | **18.63** | **0.018** | stands at every checkpoint |
| seed 2 | **G: e60, resets 4** | **120** | **27.07** | **0.959** | stands up to 60.9M, walks from 64.9M |
| seed 3 | e31 (Test 9) | 80 | 26.89 | 0.967 | stands up to 68.8M, walks only by the end |
| seed 3 | e60 (Test 9.1) | 40 | 12.78 | 0.019 | stands at every checkpoint |
| seed 3 | **C: e60, resets 5** | **80** | **26.61** (peak 27.45) | **0.971** | stands at 67.2M, walks only by the end |
| seed 3 | **D: e31, resets 20** | **40** | **18.53** | **0.019** | stands at every checkpoint |
| seed 4 | **E: e60 (default)** | 40 | 14.80 (peak 18.59 at 16.4M) | 0.018 | stands at every checkpoint |
| seed 4 | **F: e60, resets 5** | 80 | stopped at 52.4M (19.3) | not rolled out | not done |

- The two runs on the walking side of seed 2 (e31 and A) switch in the same window (49M to 52M), and the two on the walking side of seed 3 both switch late (after about 68M). So far each seed keeps its own onset when the reset interval changes between 40 and 80.
- Run G (seed 2, reset every 120) switches later, between 60.9M and 64.9M, against 49M to 52M for the two seed 2 runs at 80. This is one run with a coarser sweep, so I cannot say whether the longer interval delayed it.
- Reward on the failing side with `num_resets_per_eval` 20 ends near 18.5 to 18.6 (the standing level), not near 0 and not near 27.

## What this shows and what it does not

**Supported (in my runs):**
- The reset interval, not the number of evaluations, separates walking from standing here. Changing `num_resets_per_eval` alone turned a failing setup into a walking one (A, C) and a walking setup into a failing one (B, D), for two different seeds.
- This supports the reset-chunk reading from Test 9.1: all environments reset every 80 steps walks, every 40 stands.
- Seed 4 at 60 evaluations (default resets) also stands still, a fourth seed on the failing side.
- A reset every 120 steps (G, seed 2) also walks, so nothing breaks just above 80. With the 200-step Test 7 run, the walking side now has 80, 120 and 200.

**Not shown:**
- The Brax code reading is mine and not checked line by line. The runs agree with it, but the step arithmetic is not proof.
- Two seeds were tested in both directions. The replicate on a third seed (F) did not finish, so the fix has two seeds, not three.
- One rollout per checkpoint, one command (1 / 0 / 0).
- The intervals tried were 40, 80 and 120 steps (one seed for 120). The reset interval can only be a multiple of 40, so there is no setting between 40 and 80 and I cannot narrow the line further with `num_resets_per_eval`.
- I do not know why short chunks stop the walking. A guess is that an episode never gets past 40 steps (0.8 s), so the policy never sees what walking leads to, but I have not tested that.
- The failing runs stand and do not fall at their final checkpoints (body height about 0.3 m in the Test 9.1 rollouts). The one checkpoint I looked at in a video (D at 52.4M, below) does fall.

## Videos at 52.4M (top view)

**How this came up, in order:**
1. After the plots were made I saw a spike in the seed 3 `track` plot: run D (e31, resets 20) jumped from 0.018 to 0.032 at 52.4M, while the three other runs sat at about 0.019.
2. I asked Claude to check it. Claude read D's sweep CSV and the saved motion data: at that checkpoint the body height was negative and the upright value was -0.97, so it looked like a fall, not walking.
3. I had Claude render a top-view video of that checkpoint and watched it myself. The robot stands, tips over and ends below the floor. I guessed this was because only the feet collide, and Claude checked the model XML and confirmed it (`go1_mjx_feetonly.xml`).
4. I then had Claude render the same checkpoint for the three other seed 3 runs and watched those too. Two stand still. In C (e60, resets 5) it also stands but one hind leg twitches. Claude then compared the body shaking in the motion data and it agrees with what I saw.

I rolled out `ckpt_52428800` of the four seed 3 runs (all four have a checkpoint at this step) on 1 / 0 / 0 with the top camera, 20 s each. The videos are `top_ckpt_52428800.mp4` in each run's folder: [seed 3 e31](./go1_seed3/top_ckpt_52428800.mp4), [seed 3 e60](./go1_seed3_e60/top_ckpt_52428800.mp4), [C: e60, resets 5](./go1_t92_s3_e60_r5/top_ckpt_52428800.mp4), [D: e31, resets 20](./go1_t92_s3_e31_r20/top_ckpt_52428800.mp4).

| Run | Mean `track` | Lowest body height | Lowest upright | Moved in 20 s |
|---|---|---|---|---|
| seed 3 e31 (80) | 0.0190 | 0.288 m | 0.999 | 0.05 m |
| seed 3 e60 (40) | 0.0185 | 0.288 m | 1.000 | 0.03 m |
| C: e60, resets 5 (80) | 0.0193 | 0.287 m | 0.999 | 0.08 m |
| D: e31, resets 20 (40) | **0.0319** | **-0.347 m** | **-1.000** | 0.37 m |

- **D falls.** It stands for about 1.3 s, tips over between about 1.3 s and 2.2 s, and ends upside down with the body below the floor (about -0.26 m) for the rest of the episode. Its `track` spike of 0.032 at 52.4M in the sweep plot is this fall, not walking. Its training reward is also at the bottom of a crash there (about 1 at 52M). I do not know why it tipped; a policy in the middle of a collapse is my guess.
- **Why the body goes below the floor:** the model is `go1_mjx_feetonly.xml`, where only the four foot spheres collide with the floor (the XML says so: "enables collisions for the feet only"). The body has no collision, so a robot that flips falls through the floor and only the foot tips stay visible. The values after the fall (height, upright, `track`) are not meaningful for walking.
- **The other three stand upright** at 0.29 to 0.32 m. In C I saw a hind leg twitching in the video. The motion data agrees: C has about 40 to 50% more body shaking than the other two (gyro standard deviation 0.053 against 0.035, accelerometer 0.47 against 0.33 and 0.36) and the most movement. C is the run that walks late (after about 68M), so this could be an early sign of the switch, but it is one rollout of one checkpoint and the `track` difference (0.0193 against 0.0185 and 0.0190) is tiny.

## What went wrong with the first overnight run

I put the whole night into one chain: seven sweeps, six training runs with a sweep after each, then the plots and the copy into the repo, about 3.8 hours with one command. It went badly, even though five runs finished:
- **One big job, one point of failure.** The laptop was shut off in the morning while run F was training. The job died with it. The plots and the copy into the repo came last, so they never ran and nothing from the night was in the repo.
- **Everything stayed in WSL until the end.** Results are in the WSL folders until the final step copies them. The CSVs, logs and plots for Test 9.2 were only copied into `Tests/Test9.2/` on 2026-11-02.
- **A planned run was wrong.** The in-between run (45 evaluations) would not have shown where the line is: every `num_evals` from 31 to 59 gives an 80-step interval at 96M steps, and 45 would have run about 144M steps. I replaced it before starting.
- **Sweeps are slow.** One rollout takes about 32 s even after compiling, so I limited the sweeps to the 40M to 70M window. That window missed seed 3's switch.

**Now I split the work into small pieces:**
1. One unit is one training run plus its own sweep, about 30 minutes, started on its own in tmux.
2. Copy that run's small files into the repo and plot it straight after the unit, before the next one starts.
3. Check `go1_final` exists and the log reached the last step before moving on.
4. Check `tmux ls` and `tail ~/test92_progress.txt` before leaving, and keep the laptop plugged in and awake while a job runs.
5. Do not chain more than two or three units without checking.

## Next

1. ~~Run the plots and the copy for the five finished units.~~ Done 2026-11-02.
2. Rerun F (seed 4, 60 evaluations, `num_resets_per_eval` 5) on its own, then sweep it.
3. ~~Try one more interval.~~ Done as run G, with `num_resets_per_eval` 4, not 3: by the formula 3 would give a 160-step interval and 4 gives 120.
4. ~~Write `num_evals`, `num_resets_per_eval` and the interval in every Test setup table.~~ Done 2026-11-02 for Tests 5 to 9.1.
5. Optional: compute the body-shaking number (gyro and accelerometer spread) for every checkpoint of runs C and the seed 3 e60 run, to see whether it rises before the `track` jump.

## Evidence

- Queue script: [run_queue.sh](./run_queue.sh); run G script: [run_r4.sh](./run_r4.sh)
- Progress logs: [progress.txt](./progress.txt) (the queue, last step reached: `go1_t92_s4_e60_r5` training at 52.4M) and [progress_r4.txt](./progress_r4.txt) (run G)
- Plots (`track` and reward against steps): [all five e31 seeds](./sweep_e31_seeds.png), [seed 2 runs](./sweep_seed2.png) (including G), [seed 3 runs](./sweep_seed3.png). The seed 4 pair has no plot because F has no sweep.
- Runs A to E and G: folders `go1_t92_s2_e60_r5` (A), `go1_t92_s2_e31_r20` (B), `go1_t92_s3_e60_r5` (C), `go1_t92_s3_e31_r20` (D), `go1_t92_s4_e60` (E), `go1_t92_s2_e60_r4` (G) in this folder, each with the filtered log, reward graph, `train_test.py`, `sweep_x1.0_y0.0_yaw0.0.csv` and `sweep_out.txt`. Checkpoints stay in WSL (`~/<same folder name>`). Run F has no folder here.
- Sweeps of the earlier runs: folders `go1_seed1_e31`, `go1_seed2` to `go1_seed5`, `go1_seed2_e60`, `go1_seed3_e60` in this folder (`sweep_x1.0_y0.0_yaw0.0.csv` and `sweep_out.txt`)
- Videos at 52.4M: `top_ckpt_52428800.mp4` in the four seed 3 folders (see above)
- Scripts: [sweep.py](../../Scripts/sweep.py), [plot_sweep.py](../../Scripts/plot_sweep.py), written by Claude (see [Declaration.md](../../Declaration.md) once it is updated)
- Earlier runs: [Test 9](../Test9/Test9.md), [Test 9.1](../Test9.1/Test9.1.md)
