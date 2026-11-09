"""Plot the per-step tracking reward of one rollout (the `track` array saved in a motion_*.npz).

    python plot_track_steps.py motion_x1.5_y0.0_yaw0.0_SlopeRingsGround_1_go1_final_65s.npz -o track_steps.png

`track` is Playground's linear-velocity tracking reward for each 0.02 s step, exp(-|command - body velocity|^2 / sigma),
so 1 means the body moves exactly as commanded and 0 means it is far from it. It is only that one term of the training
reward (the full reward also has penalties for energy, falling and so on, and the rollout does not save those).
The plot shows the raw value per step (grey), a 1 s moving average (blue) and the step where the episode ended (red line).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("-o", "--out", default="track_steps.png")
args = ap.parse_args()

z = np.load(args.npz)
track = z["track"]
done = z["done"]
dt = float(z["dt"])
cmd = z["command"]
steps = np.arange(len(track))
win = int(round(1.0 / dt))  # 1 s moving average
smooth = np.convolve(track, np.ones(win) / win, mode="same")
fell = int(np.argmax(done)) if done.any() else None

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(steps, track, color="0.75", lw=0.6, label="per step")
ax.plot(steps, smooth, color="tab:blue", lw=1.8, label="1 s moving average")
if fell is not None:
    ax.axvline(fell, color="tab:red", ls="--", lw=1.2, label=f"episode ended, step {fell} ({fell * dt:.1f} s)")
ax.set_xlabel("rollout step (1 step = %.2f s)" % dt)
ax.set_ylabel("tracking reward (0 to 1)")
ax.set_ylim(-0.02, 1.05)
ax.set_title(f"Tracking reward per step, command x={cmd[0]:g}, y={cmd[1]:g}, yaw={cmd[2]:g}   (mean {track.mean():.3f})")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"mean {track.mean():.3f}; first {fell} steps mean {track[:fell].mean():.3f}" if fell else f"mean {track.mean():.3f}")
print("saved", Path(args.out))
