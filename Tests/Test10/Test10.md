# Test Log 10 — Slowest walk up the ramps, and the stairs (Test 7 policy, no new training)

**Date:** 2026-11-04
**Result:** to be added after the videos are checked.

**Purpose:** Find the slowest forward command that still gets the Test 7 policy up the first 15° ramp, then run that same command, from the same start, on a ramp terrain (`slanted`) and a stair terrain (`staired`) and record 3 camera views of each for about 60 s.

Nothing is trained in this test. The policy was only ever trained on flat ground (see [Test 7](../Test7/Test7.md)), so every slope and step here is new to it.

## Conditions (identical on both terrains)

| Setting | Value |
|---|---|
| Policy | Test 7, checkpoint `go1_final` (122,880,000 steps, flat ground), folder `~/go1_test8/go1_final` |
| Task / robot | `Go1JoystickFlatTerrain` config, `go1_mjx_feetonly.xml` (only the feet collide with the ground), no hat |
| Command (x / y / yaw) | **0.85 / 0 / 0**, held for the whole run |
| Why 0.85 | Slowest forward speed that reached the first plateau in the climb search below (0.80 did not in 80 s) |
| Length | 65 s = 3,250 steps (1 step = 0.02 s) |
| Start | Middle of the terrain, flat, z = 0, same reset for both (`PRNGKey(0)`) |
| Rollout | `Scripts/rollout_terrain.py <ground.xml> 0.85 0 0 go1_final --seconds 65` (GPU, no video) |
| Videos | `Scripts/render_qpos.py`, saved joint angles replayed on the CPU, 640 x 480, 25 fps (every second step), contact points shown |
| Cameras | `top` (looking down), `side` (fixed side view), `track` (follows the robot) |
| Run script | [run_test10.sh](./run_test10.sh), progress in `~/test10_progress.txt` |
| Hardware | NVIDIA GeForce RTX 5060 Laptop (8 GB), WSL2 Ubuntu 24.04, plugged in |

If the robot falls, the rollout stops counting it as walking (the run is marked "fell at step N") but the video still covers the full 65 s of saved states.

## The two terrains

Made by [Scripts/make_terrain.py](../../Scripts/make_terrain.py) from the layout I described, as square rings around the middle. Distance `d = max(|x|, |y|)` from the centre, in metres. The files are in `Assets/Grounds/`.

| Ring (d) | Slanted ground (`SlopeRingsGround_1`) | Staired ground (`StairRingsGround_1`) |
|---|---|---|
| 0 to 10 | flat at 0 | flat at 0 |
| 10 to 18 | ramp, 15° | stairs, 15° average (0.40 m tread, 0.107 m rise) |
| 18 to 28 | flat plateau | flat plateau |
| 28 to 35 | ramp, 25° | stairs, 25° average (0.20 m tread, 0.093 m rise) |
| 35 to 45 | flat plateau | flat plateau |
| 45 to 50 | ramp, 35° (the terrain ends at 50) | stairs, 35° average (0.15 m tread, 0.105 m rise) |

| | Slanted | Staired |
|---|---|---|
| Size | 100 x 100 m (50 m from the centre to each edge) | same |
| Grid | 0.10 m cells (1001 x 1001) | 0.05 m cells (2001 x 2001) |
| Highest point | at the edge | at the edge |

- The middle flat square is 20 x 20 m (10 m from the centre each way), which is how I read "middle 10x10 square" as a distance from the centre. Say so if you meant 10 x 10 m in total.
- A stair riser cannot be vertical on this kind of ground (a heightfield), so each riser is a steep ramp one cell (5 cm) wide.
- The stair treads are shallow so the average angle matches the ramps. They are not real building stairs.

## Step 1: finding the slowest speed that climbs

`Tests/Test10/climb_sweep.py` rolls out the same policy on the ramp terrain with command `v / 0 / 0` for 80 s, once per speed, and counts it as **arrived** when the body gets to `d >= 18.5 m` and is at least 2.0 m above the start, upright, without having fallen.

Speeds tried: 0.30, 0.50, 0.70, 0.90, 1.10, 1.30, 1.50, then 0.75, 0.80, 0.85 to narrow it down. Files: `climb_sweep_slanted.csv/.txt` and `climb_sweep_slanted_fine.csv/.txt`.

| Speed (m/s) | On first plateau | Arrived at | Farthest d (m) | Highest z (m) | Ended |
|---|---|---|---|---|---|
| 0.30 | no | - | 10.16 | 0.38 | standing / stuck at the foot of the ramp |
| 0.50 | no | - | 10.24 | 0.39 | standing / stuck |
| 0.70 | no | - | 11.12 | 0.62 | standing / stuck |
| 0.75 | no | - | 11.56 | 0.74 | standing / stuck |
| 0.80 | no | - | 16.77 | 2.13 | still climbing at 80 s, not on the plateau |
| **0.85** | **yes** | **61.3 s** | 28.13 | 2.53 | still going |
| 0.90 | yes | 44.9 s | 28.15 | 2.56 | still going |
| 1.10 | yes | 27.6 s | 28.16 | 2.56 | still going |
| 1.30 | yes | 23.3 s | 28.49 | 2.56 | fell at 78 s |
| 1.50 | yes | 21.4 s | 28.21 | 2.56 | fell at 31 s |

- No speed got past the 25° ramp (every run stopped at d of about 28 or fell), so Test 10 only compares the 15° ring and the plateau after it.
- 0.85 m/s only just makes it: it reaches the plateau at 61 s of a 65 s run, which is why the videos are 65 s and not 60 s.
- The search was done on the ramp terrain only. The stairs use the same speed, not their own minimum.

## Results

To be added: rollout numbers, graphs (`motion_*.png`, `trail_*.png`) and what the six videos show, in `slanted/` and `staired/`.

## Caveats

- One policy, one seed, one start. "Minimum speed" is the minimum among the speeds I tried (steps of 0.05 near the answer), not an exact threshold.
- Contact points in the videos are recomputed on the CPU, so they can differ slightly from the GPU simulation.
- The terrain generator, the climb search script and the run script were written by Claude from my commands (to go in Declaration.md).
