#!/bin/bash
# Test 10: the Test 7 policy on the ramp terrain (slanted) and the stair terrain (staired), same inputs, 65 s,
# command 0.85 / 0 / 0 (the slowest forward speed that got up the first 15 degree ramp, see climb_sweep_*.txt).
#   1. roll out both (GPU, no video)   2. motion graph and trail map   3. render top, side and track for both (6 renders in parallel)
# Progress: tail -f ~/test10_progress.txt
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T10=$REPO/Tests/Test10
PROGRESS=~/test10_progress.txt
V=0.85
SEC=65
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }

say "Test 10 started: command $V / 0 / 0, $SEC s, policy ~/go1_test8/go1_final (the Test 7 run)"
for pair in "slanted SlopeRingsGround_1" "staired StairRingsGround_1"; do
  set -- $pair; sub=$1; ground=$2
  mkdir -p $T10/$sub
  say "rollout $sub ($ground)"
  (cd ~/go1_test8 && python $SCRIPTS/rollout_terrain.py $REPO/Assets/Grounds/$ground.xml $V 0 0 go1_final --seconds $SEC --out $T10/$sub 2>&1 | filter > $T10/$sub/rollout_out.txt)
  tail -n 7 $T10/$sub/rollout_out.txt | tee -a $PROGRESS
done

say "motion graphs and trail maps"
for sub in slanted staired; do
  npz=$(ls $T10/$sub/motion_*.npz | head -n 1)
  python $SCRIPTS/plot_motion.py $npz -o $T10/$sub 2>&1 | filter | tail -n 2
  python $SCRIPTS/plot_trail.py $npz -o $T10/$sub 2>&1 | filter | tail -n 2
done

say "rendering 3 cameras x 2 terrains in parallel"
for pair in "slanted SlopeRingsGround_1" "staired StairRingsGround_1"; do
  set -- $pair; sub=$1; ground=$2
  npz=$(ls $T10/$sub/motion_*.npz | head -n 1)
  for cam in top side track; do
    (python $SCRIPTS/render_qpos.py $npz $REPO/Assets/Grounds/$ground.xml $T10/$sub --cams $cam 2>&1 | filter > $T10/$sub/render_$cam.txt; say "rendered $sub $cam") &
  done
done
wait
say "Test 10 finished"
