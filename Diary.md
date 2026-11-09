# Diary

The short version of [`TestingAndFinding.md`](TestingAndFinding.md). Plain words, only the numbers that matter.

> **Minimum for walking (so far, flat ground, my runs):** a reset every **80 steps or more** (every 40 steps stands still), and about **50M to 70M training steps** before the robot switches to walking, so I plan for 100M or more. Both are needed. From Test 9.2: three seeds, one rollout per checkpoint, and the reset interval is worked out from my reading of the Brax code, not measured.

---

<details>
<summary><h2>Oct 20 — I first used Colab for simulation.</h2></summary>

**Background**I haven't used colab in ages (not since that deepfake video where you can input a photo to make it sing). So I spent the afternoon refamiliarizing with the website.

**Test 1 (6M steps).** The robot took a few steps for about 8 seconds, then fell over. It was about 6million steps that is too little training.

**Test 2 (18M).** Reran only the cells I changed. The robot stayed up but stood still, and the reward looked low and strange. Probably leftover settings from my last test (a guess, not proven).

**Test 3 (18M, clean restart).** Reward jumped to 17 to 19. It looked promisng but the render result disappointed me. the robot stood still when I told it to go forward(x_vel = 1.0), and only spun on the spot when I told it to turn(yaw_vel=1.0). 

**Test 4 (25M).** Same standing robot, and the reward dropped late in training. I decided "more steps makes it worse". That was wrong.

**Late night.** I read the notebook code and found the plot cuts off the data. My "18M" and "25M" runs both really ran about 29.5M steps. Then Colab ran out of compute while I was trying 100M.

</details>

---

<details>
<summary><h2>Oct 21 — Laptop integration and the first walking not talking</h2></summary>


**Switch to Ubuntu.** I installed Ubuntu inside Windows (WSL2), Claude guided my way to get my RTX 5060 working with JAX and Warp, and moved training off Colab. The laptop ran about twice as fast as Colab's free GPU.

**Test 5 (100M steps).** About 19 minutes to train. Reward went to 26, and this time the robot **walked**: I asked for 1 m/s at direction x and it moved at about 0.96 m/s(which is close enough). So concludes the first real walking motion.

**Test 6 (control).** I needed to know the walking came from the extra steps, not from migrating to Ubuntu and the laptop. So I redid the Colab-length run (18M in the file and about 29.5M steps in the terminal) on the laptop. It stood still, just like Tests 3 and 4. So the platform wasn't the cause.

**Test 7 (120M, diagonals).** A new training run. I asked for 120M and it ran 122.9M steps (30 minutes, 16 checkpoints), and the reward ended at 26.9. It sat near 19 from 25M to 57M steps (the standing level), then climbed. I rolled the final policy out on four commands: straight (1 / 0 / 0) and three diagonals (sideways speed 0.5, -0.5 and -1). All four followed the command (`track` 0.93 to 0.95). Even sideways -1, which is outside the range it trained on (up to 0.8).

**The laptop died.** I left it running the rollouts and the battery ran out. The training had already finished and saved, so I only redid the rollouts, not the 30 minutes. Plug in before leaving.

**Test 7.5 (the odd one out).** In Test 7 the new policy walked sideways -1, but the Test 5 policy had stood still on that command. So I tried the Test 5 policy on -0.5: it walked (`track` 0.935). It can walk sideways, so the failure at -1 most likely comes from that command being out of range. Not proven, because the two runs also differ in seed, steps and evaluations.

**It still turns a little.** With the turn command at 0 the robot slowly rotated, -36 degrees over 20 seconds on the -1 diagonal. The turn command is a rate, not a heading, the reward barely notices a tiny rate, and the policy has no heading input. So it is expected, not broken. A fix would be a controller outside the policy, not retraining.

**Tools.** I put the scripts in `Scripts/`: train, rollout (it now saves the motion data), reward plot, motion graphs (distance, velocity against the command, rotation) and a bird's-eye trail map. I also looked for a chase mode. MuJoCo Playground has none, and the gait choices (bound, pace, pronk) exist for the Spot robot, not the Go1.

**Test 8 (rough terrain).** The first different ground: a bumpy rock patch. I asked for 110M steps, but the GPU ran out of memory at about 24 minutes, after 91.75M steps (82% of the plan), so the run stopped early. The checkpoints were saved, so I used the last one. The reward reached 24.1 and was still rising. The robot followed all four commands I tried: straight (`track` 0.93), the -1 diagonal (0.82), the diagonal while turning (0.91) and a slow diagonal with a fast spin of 3 (0.85). With a turn of 1 it walked three laps of a circle about 2.8 m wide. With a turn of 3, far beyond the 1.2 it trained on, it spun about nine laps of a circle about 0.5 m wide.

**Things that broke.** My first rollouts for the two circling commands crashed, because saving every step for the video filled the GPU. Moving each step to the CPU fixed it, at a cost of about 11 minutes per video. The rough patch is only 20 m wide, so a robot walking straight reaches the edge after 7 to 13 seconds. I stop the rollout there. My time guess for the training (1.5 hours) was also too high, because it ran at about 66k steps per second.

**How the robot is actually built.** I wondered what the 3D model is. It is five mesh files (STL: trunk, hip, thigh, mirrored thigh, calf) that the simulator draws, and it reuses them: one body, four hips, four thighs and four calves, so 13 pieces on screen. The pieces are joined by hinges, like door hinges: each piece can only swing around one axis, within limits, relative to the piece before it. The body itself floats free, and the 12 hinges (3 per leg) are the motors. What the simulator tracks is just 18 numbers: the body's position and tilt, and 12 joint angles. The meshes are only for the picture. The physics uses invisible simple shapes instead, like hitboxes in a game. The model has a box, cylinders and capsules for the body and legs, but they are switched off for collisions, so only the four foot spheres are live hitboxes and only they touch the ground. I also tested that a visual-only hat can be drawn on the robot without touching the physics.

**Is it like a game?** Mostly yes: rigid pieces, hinges, contacts and springs are the same ingredients as a ragdoll game, and each motor works like an elastic band at the hinge (a spring that pulls the joint towards the target angle the policy picks, with some damping). The difference is the goal. Game physics aims to look right and stay stable. MuJoCo aims to be accurate for one real robot (total mass 12.7 kg) and to run thousands of copies at once on the GPU. It still takes shortcuts: the solver does a single pass per step, which is why the screen fills with "solver iterations limit reached". MuJoCo was not made by Fan Shi. It came from Emo Todorov's group at the University of Washington, and DeepMind bought it in 2021 and made it free in 2022.

**Where walking starts.** I rolled out all 16 checkpoints of the 120M run. The robot stood still up to 57.3M steps and was walking at 65.5M, so the switch is somewhere in that gap and it is sudden. Walking was a bit slow and drifting at 65.5M, and got polished by about 80M. Reward alone had been misleading: it was already 17 at 16M steps while the robot stood still.

</details>

---

<details>
<summary><h2>Oct 22 — More seeds, and one run that collapsed</h2></summary>

**Test 9 (seeds).** I trained five more runs on flat ground (seeds 1 to 5) to see whether walking shows up reliably. Training was faster than I guessed, about 20 minutes each. Four of them (seeds 2 to 5) had reward curves like the Test 7 run: about 19 to 20 for a long stretch, then a climb that starts around 52 to 59M steps and ends at 26 to 27. I have not rolled them out yet, so I have not confirmed that they walk. Seed 1 broke. Its reward rose to 18.7 by 18M steps, dropped to about 12, and then fell to 0.2 by 85M. No checkpoint walked, and its speed stayed near zero. But seed 1 was also the only run with 60 evaluations (the others had 31), so I cannot tell whether it was the seed or the setup. In my Colab Test 4 the reward also fell late in training, and that may have been the same kind of collapse, but I do not know.

**Test 9.1 (seed or setup?).** To split the two, I trained seed 1 again with 31 evaluations, and seeds 2 and 3 with 60. Seed 1 with 31 evaluations reached 27.06, and seeds 2 and 3 with 60 stayed below 21 (they ended at 20.05 and 12.78). I rolled them out on Oct 25 (see below).

**What a seed is.** The starting number for all the randomness in training (starting network weights, random commands, noise). Same seed and settings give nearly the same run, and a different seed is different luck. Every run before Test 9 used seed 0.

**The paper.** I looked at the vision part of the paper from the lab (arXiv 2410.03076, Fan Shi is a co-author). I read the PDF text (8 pages): the vision is a very small **simulated depth image** (16 by 12 pixels, with 84 by 84 for one CNN comparison), not RGB and not a real camera. It is rendered depth rays, flattened and fed to a single dense layer, and the paper says no separate image network may be needed. The text speaks of a "field of view" but not where the camera is mounted, and it describes no real-robot experiment. The quadruped is an ANYmal C simulated in MuJoCo MJX, the same framework I use, and the navigation part uses a simple point-mass robot. So their "perception" is a low-resolution depth input to a policy trained in simulation. My Go1 policy is blind, and a small depth or height-map input would be the cheap version of the same idea.

**Accessories.** Next I want to make three weightless accessories for the robot: a pumpkin head, a cone birthday hat and a backpack. They are visual only, so they cannot change how it moves.

</details>

---

<details>
<summary><h2>Oct 25 — Seed or setup? The 60-evaluation runs stand still</h2></summary>

**Test 9.1 rollouts.** I rolled out the final policy of the three Test 9.1 runs on 1 / 0 / 0 (`--no-video`, one 20 s episode each). Seed 1 with 31 evaluations **walks** (`track` 0.968, about 0.95 m/s, 18.9 m in 20 s). Seeds 2 and 3 with 60 evaluations both **stand still** (`track` 0.0187, moved 2 to 3 cm). So seed 1 is not a broken seed. In my runs, the 60-evaluation setup is what stops the walking: all three runs with 60 evaluations failed, and the runs with 16 or 31 walked or reached 26 to 27 in reward.

**Standing, not fallen.** A fallen robot would also read near-zero speed, so I checked the saved motion data. Both 60-evaluation robots stayed at about 0.3 m body height and upright to the end. I did not watch a video.

**Reward misled me again.** Seed 2 ended at 20.05 and seed 3 at 12.78, but both give the same `track`. A low reward does not mean the robot fell, and I cannot read what the robot does from it.

**`num_evals` is not only a plotting setting.** I had assumed it only sets how often the reward is measured. Here the same seed 1 walked at 31 and collapsed at 60, so it changed the training. My guess (not tested) is that each evaluation interval is cut into chunks and all environments reset at the start of each chunk, which would make a reset every 40 steps at 60 evaluations against 80 at 31. The episode is 1000 steps, so at 60 the robot would only ever see very short pieces. The test is 60 evaluations with `num_resets_per_eval` 5. Until then, I keep `num_evals` the same for runs I compare.

**Limits.** Three runs at 60 against six at 16 or 31, one rollout each, one command, and `num_evals` also changes the random path. I have not tried values in between (such as 45).

</details>

---

<details>
<summary><h2>Oct 26 — Test 9.2: it is the reset interval, and my first overnight run went badly</h2></summary>

**What I tried.** Test 9.1 left me with "the 60-evaluation setup stops the walking". My reading of the Brax code (not checked line by line) is that all environments reset every so many steps, and that this interval depends on `num_evals` and `num_resets_per_eval`: every 40 steps at 60 evaluations, 80 at 31, 200 at 16. So I set the interval directly. I trained seed 2 and seed 3 at 60 evaluations with `num_resets_per_eval` 5 (interval 80), and at 31 evaluations with `num_resets_per_eval` 20 (interval 40). I also trained seed 4 at 60 evaluations with the default.

**What came out.** All four pairs follow the interval, not the number of evaluations. With a reset every 80 steps the robot **walks** (`track` 0.955 for seed 2, 0.971 for seed 3), and with a reset every 40 steps it **stands still** (0.018 to 0.019), whichever `num_evals` I used. Seed 4 at 60 evaluations also stands. The failing runs end at a reward of about 18.5, which is the standing level. Each seed keeps its own switch time: seed 2 switches at about 50M steps in both of its walking runs, and seed 3 switches late (after 68M) in both of its. Two seeds, both directions, one rollout each, so it is "in my runs", and I do not know why short intervals stop the walking.

**The sweeps of the old runs.** All five 31-evaluation runs walk at the end (`track` 0.952 to 0.968). The switch is sudden in each, between about 49M and 65M steps for four of them, and after 68.8M for seed 3. Seed 3's reward passed 21 at 55.7M while it still stood still on 1 / 0 / 0, so reward misled me again.

**The first overnight run went badly.** I queued about 3.8 hours of work in one chain: sweeps, six training runs with a sweep after each, then the plots and the copy into the repo at the very end. The laptop was shut off in the morning while run 6 was training (seed 4, 60 evaluations, resets 5, stopped at 52.4M of 96.7M steps). That killed the chain, and because the plots and the copy came last, nothing from the night was in the repo. Five of six runs did finish, so the science is fine, but the plan was poor. Everything lived in WSL until the end, and one bad moment cost the lot. I also had to drop my planned 45-evaluation run, because every setting from 31 to 59 evaluations gives the same 80-step interval at 96M steps, so it would not have shown where the line is.

**New rule: small pieces.** One unit is one training run plus its own sweep (about 30 minutes). After each unit I copy its small files into the repo and plot it before the next one starts, and I check that `go1_final` exists. No more than two or three units chained. Before leaving I check `tmux ls` and the progress file, and the laptop stays plugged in and awake while a job runs.

**Later (Oct 31 to Nov 2).** I plotted and copied the runs into the repo. I tried one more interval, a reset every 120 steps (seed 2, 60 evaluations, 4 resets, about 116M steps): it walks (`track` 0.959), switching a little later than the other seed 2 runs (61M to 65M against 49M to 52M, one run). I also reran the seed 4 run that died, in `Tests/Test9.2/optional_run/`: with a reset every 80 steps it walks (`track` 0.957), while the same seed at 40 steps stands, so a third seed agrees. A spike in one seed 3 plot turned out to be a fall: that run, at 52.4M steps, tipped over and went through the floor, because only the four feet collide in this model. Details are in [`Tests/Test9.2/Test9.2.md`](Tests/Test9.2/Test9.2.md).

</details>

---

<details>
<summary><h2>Oct 27 — A new ground, a party hat, and the first walk on it (Test 9.3)</h2></summary>

**What I did.** I wanted a ground I designed myself, so I described it and Claude generated it: two sine waves crossing in x and y, a 0.4 m wavelength, crest heights that vary between 1x and 1.5x, a 100 x 100 m patch and the colour `#9c7590`. It is `Assets/Grounds/SimpleUniFormRoughGround_1.xml`. The design is mine and the file was written by Claude, so it is in [`Declaration.md`](Declaration.md). I also had Claude attach my cone model (`cone_tophat.STL`) to the dog's head as a visual-only hat. It has no collisions and no mass.

**Test 9.3.** Before training on the ground I checked that it loads and that a walking policy can cross it. I used the Test 7 policy (trained on flat ground only), command 1 / 0 / 0. `track` was **0.92** on the new ground against 0.94 on flat ground, 18.5 m in 20 s, upright and no fall. Both the ground and the hat load in Playground, which I had not tried before.

**A thing that confused me.** On the stock rough terrain the same policy "fell" at 12.2 s. It did not stumble. That stock patch is only 20 m wide, and the robot walked off the edge (it crosses 10 m from the centre at about 11.8 s). Over the 612 steps before the edge its `track` was 0.887. The new ground is 100 m wide, so this cannot happen in a 20 s run.

**The hat.** I ran the same episode with the hat on. `track` was 0.9203 against 0.9199. That is inside the spread between my first two no-hat runs (0.9205 and 0.9199), so I see no effect, but it is one run, and the body turned -12.5 degrees against about -8, which I do not read as the hat.

**Videos.** Rendering on this ground is slow, about 17 minutes per camera, probably because of the 4-million-cell heightfield (I did not test that). Running the three cameras as three parallel processes took about 20 minutes in total. The README now shows the hat run, with the ground at the bottom. Details are in [`Tests/Test9.3/Test9.3.md`](Tests/Test9.3/Test9.3.md).

**Next.** Whether the ground is trainable is untested. A short smoke run (2048 environments, reset every 80 steps) would show whether it fits in 8 GB.

</details>

---

<details>
<summary><h2>Nov 4 to 7 — Ramps and stairs with the flat-ground policy (Tests 10, 10.1, 10.2)</h2></summary>

**What I did.** No training. I took the Test 7 policy (flat ground only) and put it on two new 100 x 100 m grounds I described and Claude generated: square rings with a 15° ramp, a flat plateau, a 25° ramp and so on, once as smooth ramps and once as stairs with the same average angle. It starts on the flat middle with the command held at 0.85 / 0 / 0 for 65 s.

**Finding the speed.** I first tried speeds from 0.3 to 1.5 m/s on the ramp. At 0.75 and below it stood at the foot of the ramp, 0.80 was still climbing at 80 s, and 0.85 was the slowest that reached the plateau (at 61 s). So I used 0.85 for both grounds. Nothing got past the 25° ramp.

**The ramp.** It walked up without falling. On the flat middle it did about 0.8 m/s, but on the 15° ramp only about 0.4 m/s while moving, and it stopped for 1 to 3 s several times (25 s, 38 s and 48 s). The 8 m of ramp took about 46 s, then it jumped back to 0.8 m/s on the plateau at 60.5 s. Heading stayed within 6°.

**The stairs.** It fell at the first step, 15.2 s into the run. It hit the first riser at full speed, tumbled (the heading swung about 235° in a second) and lay on its back for the other 50 s without moving. The ramp and the stairs have the same average angle, so the angle alone is not what stops it: the steps are. It has no way to see them. In the track video you only see four foot spheres after 16 s. My guess is that the trunk sank under the ground because this robot file only collides with its feet, but I did not test that.

**What I checked.** I looked at the two `track` videos as one frame every 4 s (not the `top` and `side` ones), plus the motion graphs, trail maps and numbers. Details are in [`Tests/Test10/Test10.md`](Tests/Test10/Test10.md).

**Test 10.1 (Nov 7, command 2 / 2 / 0).** I repeated the ramp run with the forward and the sideways command both raised to 2, above the trained 1.5 and 0.8. It did not get anywhere: it shook in place for about a second, tipped sideways at 1.0 s and was terminated at 1.4 s, 0.61 m from the start, on the flat middle. It lay on its back for the other 63.6 s (`track` 0.0000), and the track video only shows four foot spheres, as on the stairs. The ramp never mattered. I do not know whether x = 2 or y = 2 is the cause, because both were out of range. Details in [`Tests/Test10.1/Test10.1.md`](Tests/Test10.1/Test10.1.md).

**Test 10.2 (Nov 7, command 1.5 / 0.8 / 0).** Next I used the biggest command the policy was trained on (x 1.5, y 0.8; I checked `command_config.a = [1.5, 0.8, 1.2]` in the Playground code). This time it walked, and fast: 1.49 m/s forward and 0.75 m/s sideways on the flat middle. It crossed the 15° ramp in about 9 s (Test 10 at 0.85 took about 46 s) without stopping, then ran the plateau at full speed. At 25.5 s it reached the foot of the 25° ramp, slowed to a stop, tumbled at 26.8 s and lay on its back for the other 38 s. So 2 / 2 / 0 failed because it was out of range, not because it was fast. The 25° ramp has now beaten the policy at every command I tried. Details in [`Tests/Test10.2/Test10.2.md`](Tests/Test10.2/Test10.2.md).

**Next.** Try a faster speed on the stairs (0.85 was picked for the ramp, and the ramp is much easier at 1.5 / 0.8), split x = 2 from y = 2, and split the x and y parts of the 1.5 / 0.8 run (1.5 / 0 / 0).

</details>

---

<details open>
<summary><h2>Nov 8 — Policies trained on the ramp ground (Tests 10.3 to 10.5)</h2></summary>

**What "train on slopes" means.** Until now only Test 8 was trained on something other than flat ground (the stock rough patch). Tests 9.3 to 10.2 did not train anything: they ran the flat-trained Test 7 policy on my new grounds. Training on a ground is the same PPO and reward as before, only the ground the simulated robots stand on is different. I trained the first one on the ramp ground (`SlopeRingsGround_1`) with the Test 7 settings (120M steps, 16 evaluations, 4096 environments, seed 0). The script is `Scripts/train_terrain.py`, written by Claude.

**Training.** A 16.4M-step smoke run came first, to check the 1,000,000-cell ground fits in 8 GB (peak 4.5 GB, it did). The full run took about 26 minutes, 122.88M steps, reward 25.7 against 26.9 for the flat run, with the same plateau at 19 to 20 and a climb after about 55M steps. No out-of-memory crash this time.

**Test 10.3.** Same rollout as Test 10.2 (command 1.5 / 0.8 / 0, 65 s, ramp ground). It crossed the 15° ramp in 8.5 s (flat-trained: 9.1 s) and still failed on the 25° ramp: it walked about 0.4 m up it, slid backwards and flipped at 23.1 s (flat-trained: stalled at the foot and tumbled at 26.8 s). So training on this ground did not get it up the 25° ramp. One seed and one run, so I do not read the 3.7 s difference as worse. My guess, not tested: every training episode starts in the flat middle (within 0.5 m of the centre), so the 25° ring is rarely seen. Details in [`Tests/Test10.3/Test10.3.md`](Tests/Test10.3/Test10.3.md).

**Test 10.4 (command 1.5 / 0 / 0).** To see whether the sideways part of the command was the problem, I ran the same ramp-trained policy with y = 0. It did the same thing: 1.44 m/s on the flat, 0.86 m/s forward over the 15° ramp (10.0 s, no stops), full speed on the plateau, then it stalled at the foot of the 25° ramp at 25.0 s, crouched and toppled at 26.3 s. That is almost exactly when the flat-trained policy fell in Test 10.2 (26.8 s), so the 23.1 s of Test 10.3 looks like the odd one. Forward speed on the 15° ramp was about 0.85 m/s in all three 1.5-command runs, flat-trained or ramp-trained. The 25° ramp has now beaten every policy and command I tried. Details in [`Tests/Test10.4/Test10.4.md`](Tests/Test10.4/Test10.4.md).

**Test 10.5 (about twice the steps).** To see whether more steps help, I trained the same ramp-ground policy for 253.95M steps instead of 122.88M. The training crashed twice with a GPU out-of-memory error (after 122.88M and again 98.3M steps later, about 21 to 28 minutes into each process, with the GPU log flat at 4.5 of 8 GB), so I finished it in three parts, restoring the weights each time (the optimizer state restarts, so it is not one unbroken run). The training reward went flat after about 110M steps (26.3, 26.7, 26.7 in three windows), like the flat-ground run. With the 1.5 / 0 / 0 rollout the policy fell at the foot of the 25° ramp again (33.1 s, stalled at the foot and toppled 0.7 s after reaching it). It was also slower on the 15° ramp (0.59 m/s against 0.86), which I cannot explain from one run. After falling it kept moving its legs, which the other two never did, but it did not get up. So more steps did not help here. Details in [`Tests/Test10.5/Test10.5.md`](Tests/Test10.5/Test10.5.md).

**Next.** Measure how far training episodes actually get on this ground (the trained policy with random commands, count how many reach 10, 18 and 28 m). If the 25° ring is almost never reached, train on a ground with random start points or a growing slope.

</details>

---

<details open>
<summary><h2>What I know now</h2></summary>


- Reward isn't enough. Standing still can score about 19, and walking scored 26. I have to measure speed and watch the video.
- At about 29.5M steps the robot stands still. At about 100M it walks. (One run each, so I say "in my runs".)
- The Colab notebook wasn't broken. It was just undertrained.
- The step count in the log is the real number, not the setting. Asking for 18M gave 29.5M.
- Walking sideways and diagonally works when the command is inside the trained range (both policies).
- A command outside the range is not blocked. One policy followed it and one did not (two runs, so not a rule).
- The robot drifts in heading a little, because the turn command is a rate. The tilt is a limit of this kind of policy, not a bug.
- On rough ground the robot still followed every command I tried, even a turn of 3 (one run, stopped at 82%).
- A video rollout can run out of GPU memory. Moving each saved step to the CPU fixes it.
- The robot model is 13 mesh pieces from 5 STL files, joined by 12 hinges, with an invisible simpler shape set used for the physics. Looks and physics are separate.
- The motors act like springs at the hinges, and the solver is set to one pass per step on purpose, so the "solver iterations" warnings are normal.
- In the one run I checked with rollouts, walking started between 57.3M and 65.5M steps, suddenly. Four other seeds have reward curves that suggest about 52 to 59M, but I have not rolled them out. About 80M to 100M steps is a sensible budget, and more barely helps (the reward was flat from 98M to 123M).
- Training can collapse: seed 1's reward fell to 0.2. Test 9.1 (rolled out Oct 25) suggests it was the 60-evaluation setup and not the seed: seed 1 with 31 evaluations walks (`track` 0.968), and two other seeds with 60 evaluations stand still. Three runs against six, one rollout each, and the reason is not tested.
- `num_evals` changes the training and not only the plot, so it must stay the same for runs I compare. Test 9.2 suggests the reason is the reset interval: a reset every 80 steps walks and every 40 steps stands, whatever `num_evals` is (two seeds in both directions, and a third seed and a 120-step run agree; my reading of the Brax code, not checked line by line). Minimum for walking: a reset every 80 steps or more, plus about 50M to 70M training steps.
- Do not chain a whole night's work into one job. One unit (a run and its sweep) at a time, copied to the repo as it finishes.
- A reward of 12.8 and a reward of 20.0 can both be a robot that stands still (`track` 0.0187 for both).
- The vision in the lab's paper is a tiny simulated depth image (read from the PDF text), with no real-robot experiment described.
- The Test 7 policy, trained on flat ground only, walks across my new 100 m sine ground (`track` 0.92 against 0.94 on flat ground; one rollout, one command). Test 9.3.
- The stock rough terrain is only a 20 m patch. At 1 m/s the robot walks off its edge at about 12 s, and that is the "fall" in my rollout, not a stumble.
- A visual-only hat on the trunk made no visible difference to a rollout (`track` 0.9203 against 0.9199, one run each).
- Rendering the 4-million-cell ground costs about 17 minutes per camera. Three parallel renders took about 20 minutes in total.
- Commands well above the trained range do not give more speed: x 2 and y 2 together made the policy fall within 1.4 s on flat ground (Test 10.1, one run).
- Doubling the steps to 254M on the ramp ground did not help: the reward stayed near 26.7 after about 110M and the policy fell at the 25° ramp again (Test 10.5, one run). Long runs crash with a CUDA out-of-memory error after 21 to 28 minutes of a process, so I restart them in parts of 10 to 12 epochs.
- Training on the ramp ground (120M steps, about 26 min, 4.5 GB of GPU memory) works and reaches reward 25.7, but the policy still fell on the 25° ramp at 1.5 / 0.8 / 0 (Test 10.3, one seed) and at 1.5 / 0 / 0 (Test 10.4). It crossed the 15° ramp as fast as the flat-trained one, at about 0.85 m/s forward in all three 1.5-command runs.
- At the trained maximum (1.5 / 0.8 / 0) the policy walks at full command and crosses the 15° ramp in about 9 s, without stopping. It still falls at the foot of the 25° ramp (26.8 s, Test 10.2, one run).
- The flat-trained Test 7 policy climbs a 15° ramp only at 0.85 m/s or faster, and then slowly and in stop-and-go (about 0.4 m/s moving, 46 s for 8 m). It fell on 15° stairs at the first step (15.2 s) and never got up (Test 10, one run each, 0.85 m/s).

</details>

<details>
<summary><h2>Still don't know</h2></summary>


- Whether seeds 2 to 5 actually walk, and whether the switch lands in the same window for each (needs rollouts).
- Why a reset every 40 steps stops the walking, and where the line is between 80 and 200 steps (only 40 and 80 are tried). The seed 4 replicate of the fix did not finish.
- Whether it walks backward and turns in place (the diagonals work).
- How much a different random seed changes things, and whether the drift following the sideways command is real.
- Whether more `num_envs` (8192) changes the result.
- Whether I can train on the new ground at all: time, GPU memory (Test 8 ran out on the stock rough terrain) and whether the policy learns to walk on it. Test 9.3 only shows that a flat-trained policy walks on it.
- Whether the hat has a small effect hidden in the run-to-run spread (needs repeated runs), and whether the huge heightfield is the reason for the slow rendering.
- Whether x = 2 or y = 2 causes the fall in Test 10.1 (both were out of range), and where between 1.5 / 0.8 (walks) and 2 / 2 (falls) the policy stops following.
- Whether the faster ramp climb at 1.5 / 0.8 comes from the x part or the y part, and whether any command gets it up the 25° ramp.
- What causes the out-of-memory crash (three runs now), and why the 254M policy is slower on the 15° ramp and waves its legs after falling (one run, three resumed parts).
- How often training episodes on the ramp ground reach the 15° and 25° rings (my guess is the 25° ring almost never), and whether random start points or a slope curriculum would let the policy learn it.
- Why it falls on the stairs (the steep riser, the step height or the speed), whether another speed gets up them, and whether the trunk sinking under the ground after a fall comes from the feet-only collision file (Test 10, none of these tested).
- Whether the full 111M steps on rough ground would change the result (the reward was still rising), and why the GPU ran out of memory at epoch 14.

</details>

## Next

See [`todo.md`](todo.md) for the running list.
