"""Render the Klavuz thread region to show that the disputed face band is a third
flute land of the same helical thread, not a separate stock region."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLAVUZ_JSON = r"\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json"
OUT = r"c:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG\artifacts\klavuz_thread_lands.png"

with open(KLAVUZ_JSON) as fh:
    d = json.load(fh)
faces = sorted(d["faces"], key=lambda f: int(f["id"]))

pts = []
for f in faces:
    uv = np.asarray(f["uv"], dtype=np.float64).reshape(5, 5, 7)
    m = uv[:, :, 6] > 0.5
    p = uv[:, :, 0:3][m] if m.any() else uv[:, :, 0:3].reshape(-1, 3)
    pts.append(p)

c = np.vstack(pts).mean(axis=0)

seg1 = list(range(113, 192))
seg2 = list(range(193, 271))
disp = list(range(274, 362))
thread_all = set(seg1 + seg2 + disp + [273] + list(range(362, 365)))
other = [i for i in range(len(faces)) if i not in thread_all]

groups = [
    ("Land 1: faces 113-191  (A=Thread, GT=Thread)", seg1, "tab:green"),
    ("Land 2: faces 193-270  (A=Thread, GT=Thread)", seg2, "tab:blue"),
    ("Land 3: faces 274-361  (A=Stock, B/C/F=Thread)", disp, "tab:red"),
]

fig = plt.figure(figsize=(15, 5.5))

# 1) Unrolled cylinder: angle vs axial position.
ax = fig.add_subplot(1, 3, 1)
for name, idxs, col in groups:
    xs, ys = [], []
    for i in idxs:
        p = pts[i] - c
        z = p[:, 2]
        th = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360
        xs.append(th); ys.append(z)
    ax.scatter(np.concatenate(xs), np.concatenate(ys), s=3, c=col, label=name.split(":")[0])
ax.set_xlabel("angle around part axis (deg)")
ax.set_ylabel("axial position z")
ax.set_title("Unrolled thread surface\n3 angular lands separated by flutes")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)

# 2) Top-down view of the thread region only.
ax = fig.add_subplot(1, 3, 2)
op = np.vstack([pts[i] for i in other]) - c
ax.scatter(op[:, 0], op[:, 1], s=1, c="lightgray", label="all other faces")
for name, idxs, col in groups:
    p = np.vstack([pts[i] for i in idxs]) - c
    ax.scatter(p[:, 0], p[:, 1], s=3, c=col)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Top-down view\nlands occupy complementary sectors")
ax.grid(alpha=0.3)

# 3) Side view.
ax = fig.add_subplot(1, 3, 3)
ax.scatter(op[:, 0], op[:, 2], s=1, c="lightgray")
for name, idxs, col in groups:
    p = np.vstack([pts[i] for i in idxs]) - c
    ax.scatter(p[:, 0], p[:, 2], s=3, c=col)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("z")
ax.set_title("Side view\nidentical radius and axial span")
ax.grid(alpha=0.3)

fig.suptitle(
    "Klavuz_101 (tap): the 88 'false positive' faces are a third thread land",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
