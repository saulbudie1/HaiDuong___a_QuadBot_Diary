#!/bin/bash
# Test 10.5, training part: the Test 10.3 run again on the ramp ground (SlopeRingsGround_1), but about 2x longer.
#   250M steps requested, 32 evaluations, 4096 envs, seed 0.  32 evaluations keep the reset interval at 200 steps, the same as the
#   Test 7 and Test 10.3 runs (interval = steps per eval period / envs / resets per eval = 8,192,000 / 4096 / 10), so the only change is the length.
#   Expected: 31 epochs of 8,192,000 = 253,952,000 steps, a checkpoint every 8.2M steps, about 50 minutes.
# Progress: tail -f ~/go1_slanted_250m/train.log      GPU memory: ~/go1_slanted_250m/gpu.log
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
DIR=~/go1_slanted_250m
mkdir -p $DIR && cd $DIR
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 20 > gpu.log &
GPU=$!
python $REPO/Scripts/train_terrain.py $REPO/Assets/Grounds/SlopeRingsGround_1.xml --steps 250000000 --evals 32 --envs 4096 2>&1 | filter | tee train.log
kill $GPU
echo "max GPU memory used (MiB): $(cut -d, -f2 gpu.log | tr -dc '0-9\n' | sort -n | tail -1)"
ls $DIR | head -40
