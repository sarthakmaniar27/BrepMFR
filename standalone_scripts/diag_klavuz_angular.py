"""Angular decomposition of the Klavuz thread region.

Klavuz is a tap. A tap's helical thread is interrupted by flutes, so the thread
surface splits into several angular segments. This checks whether the disputed
band (274-361) is one of those segments.
"""

import json
from collections import Counter

import numpy as np

JSON_PATH = r"\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json"

with open(JSON_PATH) as fh:
    data = json.load(fh)
faces = sorted(data["faces"], key=lambda f: int(f["id"]))

pts_per_face = []
for f in faces:
    uv = np.asarray(f["uv"], dtype=np.float64).reshape(5, 5, 7)
    m = uv[:, :, 6] > 0.5
    p = uv[:, :, 0:3][m] if m.any() else uv[:, :, 0:3].reshape(-1, 3)
    pts_per_face.append(p)

axis = np.array([0.0, 0.0, 1.0])  # principal axis was essentially +Z
allp = np.vstack(pts_per_face)
c = allp.mean(axis=0)


def polar(idx):
    p = pts_per_face[idx] - c
    z = p @ axis
    xy = p - np.outer(z, axis)
    r = np.linalg.norm(xy, axis=1)
    th = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360.0
    return r, th, z


gt_thread = set(list(range(113, 192)) + list(range(193, 271)) + [273] + list(range(362, 365)))
disputed = set(range(274, 362))

type6 = [i for i, f in enumerate(faces) if int(f["z"]) == 6]
other_type6 = [i for i in type6 if i not in gt_thread and i not in disputed]

print(f"type-6 faces total: {len(type6)}")
print(f"  pseudo-GT thread : {len([i for i in type6 if i in gt_thread])}")
print(f"  disputed band    : {len([i for i in type6 if i in disputed])}")
print(f"  other type-6     : {len(other_type6)}")

print("\n=== angular span per segment (helical thread wraps, so span may exceed 360) ===")
segments = [
    ("GT seg 1  113-191", list(range(113, 192))),
    ("GT seg 2  193-270", list(range(193, 271))),
    ("DISPUTED  274-361", list(range(274, 362))),
    ("GT seg 4  362-364", list(range(362, 365))),
]
for tag, idxs in segments:
    rs, ths, zs = [], [], []
    for i in idxs:
        r, th, z = polar(i)
        rs.append(r); ths.append(th); zs.append(z)
    r = np.concatenate(rs); th = np.concatenate(ths); z = np.concatenate(zs)
    # Per-face mean angle, to see how faces are distributed around the axis.
    face_mean_th = []
    for i in idxs:
        _, t, _ = polar(i)
        face_mean_th.append(np.degrees(np.arctan2(np.sin(np.radians(t)).mean(),
                                                  np.cos(np.radians(t)).mean())) % 360)
    fm = np.array(face_mean_th)
    hist, _ = np.histogram(fm, bins=8, range=(0, 360))
    print(f"{tag}: n={len(idxs):3d} r={r.min():.3f}..{r.max():.3f} "
          f"z={z.min():.3f}..{z.max():.3f}  face-angle octant histogram={hist.tolist()}")

print("\n=== other type-6 faces (neither GT thread nor disputed) ===")
for i in other_type6:
    r, th, z = polar(i)
    print(f"  face {i:3d} area={float(faces[i]['y']):.5f} r={r.min():.3f}..{r.max():.3f} "
          f"z={z.min():7.3f}..{z.max():7.3f}")

print("\n=== axial (z) profile: how many thread-like faces per z slice ===")
for lo in np.arange(-0.1, 0.65, 0.05):
    hi = lo + 0.05
    def count(s):
        n = 0
        for i in s:
            _, _, z = polar(i)
            if ((z >= lo) & (z < hi)).any():
                n += 1
        return n
    print(f"  z {lo:6.2f}..{hi:6.2f}  gt_thread={count(gt_thread):3d}  disputed={count(disputed):3d}")
