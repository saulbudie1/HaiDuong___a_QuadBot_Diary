#!/bin/bash
# Test 10.5, part 3: part 2 (train_250M_part2.sh) also crashed with a Warp CUDA out-of-memory error, after 98,304,000 of its steps
# (see train_part2_crashed.log). Resume from its last checkpoint, ckpt_98304000, for 4 more epochs of 8,192,000 = 32,768,000 steps
# (5 evaluations, same epoch size, so the reset interval stays 200). A short process, so it should finish before the crash.
# Total: 122,880,000 (part 1) + 98,304,000 (part 2) + 32,768,000 (part 3) = 253,952,000 steps, as the original 250M run would have run.
# Only the weights are restored each time (optimizer state and step counter restart), and I use seed 2 here.
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
DIR=~/go1_slanted_250m_part3
mkdir -p $DIR && cd $DIR
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 20 > gpu.log &
GPU=$!
python $REPO/Scripts/train_terrain.py $REPO/Assets/Grounds/SlopeRingsGround_1.xml --steps 32768000 --evals 5 --envs 4096 --seed 2 --resume ~/go1_slanted_250m_part2/ckpt_98304000 2>&1 | filter | tee train.log
kill $GPU
echo "max GPU memory used (MiB): $(cut -d, -f2 gpu.log | tr -dc '0-9\n' | sort -n | tail -1)"
ls $DIR | head -20
