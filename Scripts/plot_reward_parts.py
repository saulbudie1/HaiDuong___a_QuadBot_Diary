"""Plot the eval reward of one training run that was done in several resumed parts, as one line, next to other runs.

    python plot_reward_parts.py -o reward_plot.png \
        --run "Test 10.5 (3 parts)" part1.log part2.log part3.log \
        --run "Test 10.3" train.log --run "Test 7, flat" run8.log

Each --run is a label followed by one or more `steps=... reward=...` logs (the lines train_test.py / train_terrain.py print).
The logs of a run are joined in the order given: the step numbers of every part after the first are shifted by the last step
of the part before it (a resumed part counts its steps from 0), and the first line of a resumed part (step 0 = the evaluation of
the restored weights, before any training) is dropped. The shifted joins are marked with thin vertical lines.
Only the weights are restored between parts (the optimizer state restarts), so the joined line is not one unbroken run.
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"steps=\s*(\d+)\s+reward=(-?[\d.]+)")


def read(path):
    return [(int(m.group(1)), float(m.group(2))) for m in map(LINE.search, Path(path).read_text(errors="replace").splitlines()) if m]


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", default="reward_plot.png")
ap.add_argument("--run", nargs="+", action="append", metavar=("LABEL", "LOG"), required=True)
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(10, 5.2))
for label, *logs in args.run:
    xs, ys, offset, joins = [], [], 0, []
    for i, log in enumerate(logs):
        rows = read(log)
        if i > 0:
            rows = rows[1:]  # step 0 of a resumed part is the evaluation of the restored weights, already counted
            joins.append(offset)
        for step, reward in rows:
            xs.append((step + offset) / 1e6)
            ys.append(reward)
        offset += rows[-1][0]
    line, = ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, label=f"{label}  (final {ys[-1]:.2f} at {xs[-1]:.1f}M)")
    for j in joins:
        ax.axvline(j / 1e6, color=line.get_color(), lw=0.8, ls=":", alpha=0.7)
ax.set_xlabel("environment steps (millions)")
ax.set_ylabel("eval episode reward")
ax.set_title("Go1 joystick: eval reward vs steps (dotted lines = where a run was resumed)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print("saved", args.out)
