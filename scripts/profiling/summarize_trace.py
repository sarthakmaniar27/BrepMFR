import json, sys
from collections import defaultdict

def main(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    durations = defaultdict(float)
    counts = defaultdict(int)
    for ev in events:
        if ev.get("ph") != "X":  # complete events only
            continue
        name = ev.get("name", "")
        dur_us = float(ev.get("dur", 0.0))
        durations[name] += dur_us
        counts[name] += 1

    top = sorted(durations.items(), key=lambda x: x[1], reverse=True)[:40]
    print(f"Events with duration (ph=X): {sum(counts.values())}")
    for name, us in top:
        ms = us / 1000.0
        print(f"{ms:10.2f} ms  (n={counts[name]:6d})  {name[:120]}")

if __name__ == "__main__":
    main(sys.argv[1])