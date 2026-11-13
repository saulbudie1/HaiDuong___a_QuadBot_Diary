#!/bin/bash
# Test 12: Test 11 again (SlopeRingsGround_2, ramp 2 m from the start, 120M steps requested, 16 evaluations, 4096 envs, seed 0)
# with ONE change: --resets 2, so all robots are reset every 1000 steps (20 s, the episode cap) instead of every 200 steps (4 s).
# Then roll out 1.5 / 0 / 0 for 65 s on the new ground and on the old ground SlopeRingsGround_1 (numbers and plots only).
# Progress: tail -f ~/test12_progress.txt
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T=$REPO/Tests/Test12
G1=$REPO/Assets/Grounds/SlopeRingsGround_1.xml
G2=$REPO/Assets/Grounds/SlopeRingsGround_2.xml
PROGRESS=~/test12_progress.txt
DIR=~/go1_long
X=1.5; Y=0; SEC=65
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }
: > $PROGRESS
mkdir -p $T/near $T/orig $T/policy
cp "$0" $T/run_test12.sh

say "Test 12 started"
say "training on SlopeRingsGround_2: 120M requested, 16 evaluations, 4096 envs, seed 0, --resets 2"
mkdir -p $DIR && cd $DIR
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 20 > gpu.log &
GPU=$!
python $SCRIPTS/train_terrain.py $G2 --steps 120000000 --evals 16 --envs 4096 --resets 2 2>&1 | filter | tee train.log
kill $GPU
if [ ! -e $DIR/go1_final ]; then
  say "TRAINING DID NOT FINISH (no go1_final). Last lines:"; tail -n 5 train.log | tee -a $PROGRESS; exit 1
fi
say "training finished"; grep -E "num_resets|steps=|time to|last step" train.log | tail -n 20 | tee -a $PROGRESS
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
say "Test 12 finished"
