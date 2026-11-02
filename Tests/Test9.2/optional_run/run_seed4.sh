#!/bin/bash
# Test 9.2 optional run (replicate F): seed 4, 60 evaluations, num_resets_per_eval 5 (reset every 80 steps)
# 1. train  2. sweep 40M to 70M  3. top-view videos (52.4M and final)  4. plot against run E  5. copy into Tests/Test9.2/optional_run
source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
DEST=$REPO/Tests/Test9.2/optional_run
PROGRESS=~/test92_seed4_progress.txt
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }

name=go1_t92_s4_e60_r5_rerun
mkdir -p $DEST
say "train $name (seed 4, evals 60, resets 5, steps 96_000_000)"
mkdir -p ~/$name && cd ~/$name || exit 1
tr -d '\r' < $SCRIPTS/train_test.py | sed \
  -e "s/^ppo_params.num_timesteps = .*/ppo_params.num_timesteps = 96_000_000/" \
  -e "s/^ppo_params.num_evals = .*/ppo_params.num_evals = 60/" \
  -e "s/^    policy_params_fn=save_ckpt,\$/&\n    seed=4,/" > train_test.py
sed -i "s/^ppo_params.num_envs = .*/&\nppo_params.num_resets_per_eval = 5/" train_test.py
grep -nE "num_timesteps|num_evals|num_resets|seed=" train_test.py | tee -a $PROGRESS
python train_test.py 2>&1 | filter > run_$name.log
tail -n 4 run_$name.log | tee -a $PROGRESS
python $SCRIPTS/plot_reward.py run_$name.log -o reward_plot.png > /dev/null

if [ ! -e go1_final ]; then say "no go1_final, training did not finish"; fi

say "sweep $name"
python $SCRIPTS/sweep.py 1 0 0 --min-step 40e6 --max-step 70e6 --every 2 2>&1 | filter > sweep_out.txt
tail -n 3 sweep_out.txt | tee -a $PROGRESS

say "videos (top view)"
cp /mnt/c/Users/Admin/AppData/Local/Temp/claude/c--Users-Admin-Documents-HaiDuong/f8098c25-3c3c-40a5-a3bf-29a2a28c4e55/scratchpad/rollout_top.py ~/rollout_top.py
for ck in ckpt_52428800 go1_final; do
  [ -e $ck ] || { say "missing $ck"; continue; }
  python ~/rollout_top.py 1 0 0 $ck 2>&1 | grep -E "mean track" | tee -a $PROGRESS
done

say "plot and copy"
cd ~
python $SCRIPTS/plot_sweep.py go1_t92_s4_e60 $name -o $DEST/sweep_seed4.png
cd ~/$name
cp run_$name.log reward_plot.png train_test.py sweep_x1.0_y0.0_yaw0.0.csv sweep_out.txt $DEST/
cp top_*.mp4 $DEST/
cp ~/run_seed4.sh $DEST/run_seed4.sh
cp $PROGRESS $DEST/progress.txt
say "finished"
cp $PROGRESS $DEST/progress.txt
