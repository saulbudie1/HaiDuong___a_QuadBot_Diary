#!/bin/bash
# Test 11.2: the Test 11 policy trained for another 122,880,000 steps, about 245.76M in total (Test 11: 122.88M).
# Same ground (SlopeRingsGround_2, ramp 2 m from the start), same settings as Test 11 (16 evaluations, 4096 envs, the default reset
# interval of 200 steps). Resume from ~/go1_near/ckpt_122880000 (the last checkpoint of Test 11, the same weights as go1_final) in a
# fresh process and a new folder, as Test 10.5 did (a long run in one process crashed there with a CUDA out-of-memory error after 21 to 28
# minutes). Only the weights are restored (the optimizer state and the step counter restart, see train_terrain.py --resume), and I use
# seed 1 so the second part does not replay the random numbers of the first, as in Test 10.5 part 2.
# Then roll out 1.5 / 0 / 0 for 65 s on the new ground and on the old ground, plot, and render the three cameras of the new-ground rollout.
# Progress: tail -f ~/test11_2_progress.txt      step numbers in ~/go1_near_250m/train.log count from 0 (add 122,880,000)
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
T=$REPO/Tests/Test11.2
G1=$REPO/Assets/Grounds/SlopeRingsGround_1.xml
G2=$REPO/Assets/Grounds/SlopeRingsGround_2.xml
PROGRESS=~/test11_2_progress.txt
DIR=~/go1_near_250m
X=1.5; Y=0; SEC=65
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }
: > $PROGRESS
mkdir -p $T/near $T/orig $T/policy
cp "$0" $T/run_test11_2.sh

say "Test 11.2 started"
say "training on SlopeRingsGround_2: another 122,880,000 steps from ~/go1_near/ckpt_122880000, 16 evaluations, 4096 envs, seed 1"
mkdir -p $DIR && cd $DIR
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 20 > gpu.log &
GPU=$!
python $SCRIPTS/train_terrain.py $G2 --steps 122880000 --evals 16 --envs 4096 --seed 1 --resume ~/go1_near/ckpt_122880000 2>&1 | filter | tee train.log
kill $GPU
if [ ! -e $DIR/go1_final ]; then
  say "TRAINING DID NOT FINISH (no go1_final). Last lines and checkpoints:"; tail -n 5 train.log | tee -a $PROGRESS; ls $DIR | grep ckpt | tail -n 3 | tee -a $PROGRESS; exit 1
fi
say "training finished"; grep -E "num_resets|steps=|time to|last step" train.log | tail -n 20 | tee -a $PROGRESS
echo "max GPU memory used (MiB): $(cut -d, -f2 gpu.log | tr -dc '0-9\n' | sort -n | tail -1)" | tee -a $PROGRESS
cp go1_final $T/policy/go1_final
cp train.log gpu.log $T/

# one reward curve for the whole 245.76M: the Test 11 log, then this log with the step numbers shifted by 122,880,000
python - <<'EOF'
import re, os
home = os.path.expanduser("~")
pat = re.compile(r"(steps=\s*)(\d+)(\s+reward=.*)")
out = []
for line in open(home + "/go1_near/train.log", errors="replace"):
    if pat.search(line): out.append(line.rstrip("\n"))
for line in open(home + "/go1_near_250m/train.log", errors="replace"):
    m = pat.search(line)
    if m and int(m[2]) > 0:
        out.append(f"{m[1]}{int(m[2]) + 122880000:>10}{m[3]}")
open(home + "/go1_near_250m/train_combined.log", "w").write("\n".join(out) + "\n")
EOF
cp train_combined.log $T/
python $SCRIPTS/plot_reward.py train_combined.log -o $T/reward_plot.png 2>&1 | filter | tail -n 24

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
say "numbers and plots done"

say "rendering 3 cameras of the new-ground rollout in parallel (about 20 min)"
npz=$(ls $T/near/motion_*.npz | head -n 1)
for cam in top side track; do
  (python $SCRIPTS/render_qpos.py $npz $G2 $T/near --cams $cam 2>&1 | filter > $T/near/render_$cam.txt; say "rendered near $cam") &
done
wait
say "Test 11.2 finished"
