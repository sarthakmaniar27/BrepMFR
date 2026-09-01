"""Check the edge_ang wrap discontinuity.

json_to_brepmfr_pyg wraps the dihedral angle with (a + pi) % (2*pi) - pi and then
feeds the result to the network as a RAW SCALAR through NonLinear(1, num_heads).
An angle is circular, so +pi and -pi are the same geometry but the most distant
possible scalar inputs. This measures how many Klavuz thread edges sit on that
discontinuity.
"""

import json
from collections import Counter

import numpy as np

KLAVUZ_JSON = r"\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json"
PI = np.pi
TWO_PI = 2 * np.pi


def wrap(a):
    return (a + PI) % TWO_PI - PI


with open(KLAVUZ_JSON) as fh:
    d = json.load(fh)

thread = set(
    list(range(113, 192)) + list(range(193, 271)) + [273]
    + list(range(274, 362)) + list(range(362, 365))
)

raw_thread, raw_all = [], []
for e in d["edges"]:
    a = float(e["a"])
    raw_all.append(a)
    nf = e.get("nf", [])
    if len(nf) == 2 and int(nf[0]) in thread and int(nf[1]) in thread:
        raw_thread.append(a)

raw_thread = np.array(raw_thread)
raw_all = np.array(raw_all)
w_thread = wrap(raw_thread)
w_all = wrap(raw_all)

print(f"Klavuz edges total: {len(raw_all)}   thread-thread edges: {len(raw_thread)}")
print("\n--- RAW dihedral angle on thread-thread edges ---")
print(f"  min={raw_thread.min():.9f}  median={np.median(raw_thread):.9f}  max={raw_thread.max():.9f}")
print(f"  count raw >  pi : {(raw_thread > PI).sum()}  ({100*(raw_thread>PI).mean():.1f}%)")
print(f"  count raw == pi within 1e-6 : {(np.abs(raw_thread - PI) < 1e-6).sum()}")
print(f"  count |raw - pi| < 1e-3     : {(np.abs(raw_thread - PI) < 1e-3).sum()}")

print("\n--- AFTER wrap((a+pi) % 2pi - pi) ---")
print(f"  min={w_thread.min():.9f}  median={np.median(w_thread):.9f}  max={w_thread.max():.9f}")
neg = (w_thread < -3.0).sum()
pos = (w_thread > 3.0).sum()
print(f"  landing near -pi (< -3.0): {neg}  ({100.0*neg/len(w_thread):.1f}%)")
print(f"  landing near +pi (> +3.0): {pos}  ({100.0*pos/len(w_thread):.1f}%)")
print("  --> geometrically identical flat edges split across the two extremes"
      if neg and pos else
      "  --> all flat thread edges collapse to one extreme")

print("\n--- histogram of wrapped angle, ALL Klavuz edges ---")
h, edges = np.histogram(w_all, bins=12, range=(-PI, PI))
for i in range(12):
    print(f"  [{edges[i]:6.2f},{edges[i+1]:6.2f}) : {h[i]:5d}")

print("\n--- how many ALL-edge raw values exceed pi (i.e. get wrapped) ---")
print(f"  raw > pi : {(raw_all > PI).sum()} / {len(raw_all)}  "
      f"({100.0*(raw_all>PI).mean():.1f}%)")
print(f"  raw max  : {raw_all.max():.9f}   (2*pi = {TWO_PI:.9f})")
print(f"  raw min  : {raw_all.min():.9f}")

print("\n--- raw value multiset near pi (top 5 most common raw angles) ---")
c = Counter(np.round(raw_all, 6).tolist())
for v, n in c.most_common(8):
    print(f"  raw={v:12.6f}  n={n:5d}   wrapped={wrap(v):12.6f}")
