#!/bin/bash
# Test 11.1, videos: waits for run_test11_1.sh to finish (it prints "Test 11.1 finished" in ~/test11_1_progress.txt), then renders
# the three cameras (top, side, track) of the new-ground rollout in parallel, as Test 11 did (about 20 min).
# Kept as its own script because run_test11_1.sh was already running when the videos were asked for.
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T=$REPO/Tests/Test11.1
G2=$REPO/Assets/Grounds/SlopeRingsGround_2.xml
PROGRESS=~/test11_1_progress.txt
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }
cp "$0" $T/render_test11_1.sh

until grep -qE "Test 11.1 finished|TRAINING DID NOT FINISH" $PROGRESS 2>/dev/null; do sleep 30; done
if grep -q "TRAINING DID NOT FINISH" $PROGRESS; then say "no videos: training did not finish"; exit 1; fi

say "rendering 3 cameras of the new-ground rollout in parallel (about 20 min)"
npz=$(ls $T/near/motion_*.npz | head -n 1)
for cam in top side track; do
  (python $SCRIPTS/render_qpos.py $npz $G2 $T/near --cams $cam 2>&1 | filter > $T/near/render_$cam.txt; say "rendered near $cam") &
done
wait
say "Test 11.1 videos finished"
