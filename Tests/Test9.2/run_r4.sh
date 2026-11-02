#!/bin/bash
# Test 9.2 follow-up: seed 2, 60 evaluations, num_resets_per_eval 4 (reset every 120 steps), then sweep
source ~/go1env/bin/activate
SCRIPTS=/mnt/c/Users/Admin/Documents/HaiDuong/Scripts
PROGRESS=~/test92_r4_progress.txt
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }

name=go1_t92_s2_e60_r4
say "train $name (seed 2, evals 60, resets 4, steps 96_000_000)"
mkdir -p ~/$name && cd ~/$name || exit 1
tr -d '\r' < $SCRIPTS/train_test.py | sed \
  -e "s/^ppo_params.num_timesteps = .*/ppo_params.num_timesteps = 96_000_000/" \
  -e "s/^ppo_params.num_evals = .*/ppo_params.num_evals = 60/" \
  -e "s/^    policy_params_fn=save_ckpt,\$/&\n    seed=2,/" > train_test.py
sed -i "s/^ppo_params.num_envs = .*/&\nppo_params.num_resets_per_eval = 4/" train_test.py
grep -nE "num_timesteps|num_evals|num_resets|seed=" train_test.py | tee -a $PROGRESS
python train_test.py 2>&1 | filter > run_$name.log
tail -n 4 run_$name.log | tee -a $PROGRESS
python $SCRIPTS/plot_reward.py run_$name.log -o reward_plot.png > /dev/null

say "sweep $name"
python $SCRIPTS/sweep.py 1 0 0 --min-step 40e6 --max-step 70e6 --every 2 2>&1 | filter > sweep_out.txt
tail -n 3 sweep_out.txt | tee -a $PROGRESS
say "finished"
