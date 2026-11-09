#!/bin/bash
# Test 10.5, part 2: the first part (train_250M.sh) crashed after 122,880,000 steps with a Warp CUDA out-of-memory error
# (see train_part1_crashed.log). Resume from its last checkpoint, ckpt_122880000, in a fresh process and a new folder, for another
# 122,880,000 steps (16 evaluations = 15 epochs of 8,192,000, the same epoch size, so the reset interval stays 200 steps).
# Total about 245.76M steps. Only the weights are restored (optimizer state and step counter restart, see train_terrain.py --resume),
# and I use seed 1 so the second part does not replay the random numbers of the first.
# Progress: tail -f ~/go1_slanted_250m_part2/train.log      step numbers in this log count from 0 (add 122,880,000)
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
DIR=~/go1_slanted_250m_part2
mkdir -p $DIR && cd $DIR
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 20 > gpu.log &
GPU=$!
python $REPO/Scripts/train_terrain.py $REPO/Assets/Grounds/SlopeRingsGround_1.xml --steps 122880000 --evals 16 --envs 4096 --seed 1 --resume ~/go1_slanted_250m/ckpt_122880000 2>&1 | filter | tee train.log
kill $GPU
echo "max GPU memory used (MiB): $(cut -d, -f2 gpu.log | tr -dc '0-9\n' | sort -n | tail -1)"
ls $DIR | head -40
