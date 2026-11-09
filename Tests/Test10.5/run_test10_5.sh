#!/bin/bash
# Test 10.5: Test 10.4 again (ramp ground, command 1.5 / 0 / 0, 65 s), but with the policy trained for 253,952,000 steps (Test 10.5 training, 3 parts; Test 10.4: 122,880,000).
# Policy: ~/go1_slanted_250m_part3/go1_final (the last of three resumed parts, see train_250M*.sh).
#   1. rollout (GPU, no video)   2. motion graph and trail map   3. render top, side and track (3 renders in parallel)
# Progress: tail -f ~/test10_5_progress.txt
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T=$REPO/Tests/Test10.5
PROGRESS=~/test10_5_progress.txt
GROUND=SlopeRingsGround_1
X=1.5; Y=0; SEC=65
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }

say "Test 10.5 started: command $X / $Y / 0, $SEC s, policy ~/go1_slanted_250m_part3/go1_final (trained on this ground), $GROUND"
mkdir -p $T/slanted
(cd ~/go1_slanted_250m_part3 && python $SCRIPTS/rollout_terrain.py $REPO/Assets/Grounds/$GROUND.xml $X $Y 0 go1_final --seconds $SEC --out $T/slanted 2>&1 | filter > $T/slanted/rollout_out.txt)
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
say "Test 10.5 finished"
