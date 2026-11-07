#!/bin/bash
# Test 10.3, training part: train the Go1 on the ramp ground (SlopeRingsGround_1) in ~/go1_slanted.
#   bash train_slanted.sh smoke   -> 16.4M steps, only to check that it fits in 8 GB, how fast it is, and that it starts
#   bash train_slanted.sh full    -> 120M steps, 16 evaluations, 4096 envs (the Test 7 settings, so the runs are comparable)
# Progress: tail -f ~/go1_slanted/train.log      GPU memory: ~/go1_slanted/gpu.log
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
MODE=${1:-smoke}
DIR=~/go1_slanted
[ "$MODE" = smoke ] && DIR=~/go1_slanted_smoke
mkdir -p $DIR && cd $DIR
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations|^Module .* load on device|Failed to all|Source Location|external/xla|^\s*$"; }
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 20 > gpu.log &
GPU=$!
if [ "$MODE" = smoke ]; then
  python $REPO/Scripts/train_terrain.py $REPO/Assets/Grounds/SlopeRingsGround_1.xml --steps 16000000 --evals 3 --envs 4096 2>&1 | filter | tee train.log
else
  python $REPO/Scripts/train_terrain.py $REPO/Assets/Grounds/SlopeRingsGround_1.xml --steps 120000000 --evals 16 --envs 4096 2>&1 | filter | tee train.log
fi
kill $GPU
echo "max GPU memory used (MiB): $(cut -d, -f2 gpu.log | tr -dc '0-9\n' | sort -n | tail -1)"
ls $DIR | head -30
