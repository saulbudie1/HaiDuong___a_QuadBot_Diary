#!/bin/bash
# Test 10.3: Test 10.2 again, but with a policy trained on the ramp ground (SlopeRingsGround_1) itself, see train_slanted.sh.
# Same ground, command 1.5 / 0.8 / 0, start, length (65 s) and cameras as Test 10.2.
#   1. rollout (GPU, no video)   2. motion graph and trail map   3. render top, side and track (3 renders in parallel)
# Progress: tail -f ~/test10_3_progress.txt
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T=$REPO/Tests/Test10.3
PROGRESS=~/test10_3_progress.txt
GROUND=SlopeRingsGround_1
X=1.5; Y=0.8; SEC=65
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }

say "Test 10.3 started: command $X / $Y / 0, $SEC s, policy ~/go1_slanted/go1_final (trained on this ground), $GROUND"
mkdir -p $T/slanted
(cd ~/go1_slanted && python $SCRIPTS/rollout_terrain.py $REPO/Assets/Grounds/$GROUND.xml $X $Y 0 go1_final --seconds $SEC --out $T/slanted 2>&1 | filter > $T/slanted/rollout_out.txt)
tail -n 7 $T/slanted/rollout_out.txt | tee -a $PROGRESS

say "motion graph and trail map"
npz=$(ls $T/slanted/motion_*.npz | head -n 1)
python $SCRIPTS/plot_motion.py $npz -o $T/slanted 2>&1 | filter | tail -n 2
python $SCRIPTS/plot_trail.py $npz -o $T/slanted 2>&1 | filter | tail -n 2

say "rendering 3 cameras in parallel"
for cam in top side track; do
  (python $SCRIPTS/render_qpos.py $npz $REPO/Assets/Grounds/$GROUND.xml $T/slanted --cams $cam 2>&1 | filter > $T/slanted/render_$cam.txt; say "rendered slanted $cam") &
done
wait
say "Test 10.3 finished"
