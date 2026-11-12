#!/bin/bash
# Test 11: train on SlopeRingsGround_2 (SlopeRingsGround_1 with every band 8 m closer, so the 15 degree ramp starts 2 m from the
# start point) with the standard settings (120M steps requested, 16 evaluations, 4096 envs, seed 0, the same as Test 7 and 10.3),
# then roll out 1.5 / 0 / 0 for 65 s (the Test 10.4 and 10.5 rollout) on the new ground (with videos) and on the old ground
# SlopeRingsGround_1 (numbers and plots only), so the 25 degree ramp can be compared with Tests 10.4 and 10.5.
# Progress: tail -f ~/test11_progress.txt
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T=$REPO/Tests/Test11
G1=$REPO/Assets/Grounds/SlopeRingsGround_1.xml
G2=$REPO/Assets/Grounds/SlopeRingsGround_2.xml
PROGRESS=~/test11_progress.txt
DIR=~/go1_near
X=1.5; Y=0; SEC=65
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }
: > $PROGRESS
mkdir -p $T/near $T/orig $T/policy
cp "$0" $T/run_test11.sh

say "Test 11 started"
python $SCRIPTS/make_terrain.py --near 2>&1 | tee -a $PROGRESS

say "training on SlopeRingsGround_2: 120M requested, 16 evaluations, 4096 envs, seed 0"
mkdir -p $DIR && cd $DIR
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 20 > gpu.log &
GPU=$!
python $SCRIPTS/train_terrain.py $G2 --steps 120000000 --evals 16 --envs 4096 2>&1 | filter | tee train.log
kill $GPU
if [ ! -e $DIR/go1_final ]; then
  say "TRAINING DID NOT FINISH (no go1_final). Last lines:"; tail -n 5 train.log | tee -a $PROGRESS; exit 1
fi
say "training finished"; grep -E "steps=|time to|last step" train.log | tail -n 20 | tee -a $PROGRESS
echo "max GPU memory used (MiB): $(cut -d, -f2 gpu.log | tr -dc '0-9\n' | sort -n | tail -1)" | tee -a $PROGRESS
cp go1_final $T/policy/go1_final
cp train.log gpu.log $T/
python $SCRIPTS/plot_reward.py train.log -o $T/reward_plot.png 2>&1 | filter | tail -n 3

say "rollout on the new ground (SlopeRingsGround_2), $X / $Y / 0, $SEC s"
python $SCRIPTS/rollout_terrain.py $G2 $X $Y 0 go1_final --seconds $SEC --out $T/near 2>&1 | filter > $T/near/rollout_out.txt
tail -n 7 $T/near/rollout_out.txt | tee -a $PROGRESS
say "rollout on the old ground (SlopeRingsGround_1), same command"
python $SCRIPTS/rollout_terrain.py $G1 $X $Y 0 go1_final --seconds $SEC --out $T/orig 2>&1 | filter > $T/orig/rollout_out.txt
tail -n 7 $T/orig/rollout_out.txt | tee -a $PROGRESS

say "motion graphs, trail maps, tracking-reward plots"
for d in near orig; do
  npz=$(ls $T/$d/motion_*.npz | head -n 1)
  python $SCRIPTS/plot_motion.py $npz -o $T/$d 2>&1 | filter | tail -n 2
  python $SCRIPTS/plot_trail.py $npz -o $T/$d 2>&1 | filter | tail -n 2
  python $SCRIPTS/plot_track_steps.py $npz -o $T/$d/track_steps.png 2>&1 | filter | tail -n 2
done

say "rendering 3 cameras of the new-ground rollout in parallel (about 20 min)"
npz=$(ls $T/near/motion_*.npz | head -n 1)
for cam in top side track; do
  (python $SCRIPTS/render_qpos.py $npz $G2 $T/near --cams $cam 2>&1 | filter > $T/near/render_$cam.txt; say "rendered near $cam") &
done
wait
say "Test 11 finished"
