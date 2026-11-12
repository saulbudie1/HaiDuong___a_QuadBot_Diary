import numpy as np
z = np.load("motion_x1.0_y0.0_yaw0.0_go1_final.npz")
t = z["track"]
print(np.round(t[:120], 2))
print(np.argmax(t > 0.9))
print(t.max())