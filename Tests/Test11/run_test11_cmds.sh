#!/bin/bash
# Test 11, cheap check 2: the same policy (~/go1_near/go1_final) on the same new ground (SlopeRingsGround_2), 65 s, start in the
# middle, PRNGKey(0), at commands 1.0 / 0 / 0 and 0.85 / 0 / 0 instead of 1.5 / 0 / 0. GPU rollouts only, no video.
# Does the freeze on the 15 degree ramp move or go away when the command changes?
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T=$REPO/Tests/Test11/near
G2=$REPO/Assets/Grounds/SlopeRingsGround_2.xml
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
cp "$0" $REPO/Tests/Test11/run_test11_cmds.sh
cd ~/go1_near
for X in 1.0 0.85; do
  echo "[$(date +%H:%M)] rollout $X / 0 / 0"
  python $SCRIPTS/rollout_terrain.py $G2 $X 0 0 go1_final --seconds 65 --out $T 2>&1 | filter > $T/rollout_out_x$X.txt
  tail -n 7 $T/rollout_out_x$X.txt
  npz=$(ls $T/motion_x${X}_*SlopeRingsGround_2*.npz | head -n 1)
  python $SCRIPTS/plot_motion.py $npz -o $T 2>&1 | filter | tail -n 1
  python $SCRIPTS/plot_trail.py $npz -o $T 2>&1 | filter | tail -n 1
  python $SCRIPTS/plot_track_steps.py $npz -o $T/track_steps_x$X.png 2>&1 | filter | tail -n 1
done
echo "[$(date +%H:%M)] done"
