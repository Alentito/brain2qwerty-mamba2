"""Score all completed models directly on the Kelvin-2 cluster."""

import json
from pathlib import Path

CHAR_INDEX = {
    0: "s", 1: "o", 2: "t", 3: "e", 4: "n", 5: "c", 6: "i", 7: "a",
    8: " ", 9: "d", 10: "l", 11: "r", 12: "b", 13: "@", 14: "z",
    15: "v", 16: "f", 17: "m", 18: "u", 19: "h", 20: "p", 21: "g",
    22: "q", 23: "w", 24: "x", 25: "y", 26: "j", 27: "k", 28: "9"
}

def lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def main():
    search_dirs = [
        Path.home() / "sharedscratch/B2Q/cache_v1mamba/results",
        Path("results"),
        Path.home() / ".cache/b2q_v1mamba/results",
        Path("checkpoints"),
    ]
    
    results_dir = None
    for d in search_dirs:
        if d.exists() and list(d.glob("*/callbacks/test_all_sentences.json")):
            results_dir = d
            break
            
    if not results_dir:
        print("No results directories found with test_all_sentences.json.")
        return

    print("\n" + "="*85)
    print(f"{'Model Name':<58} {'Test CER':>10} {'Sentences':>12}")
    print("="*85)

    rows = []
    for p in sorted(results_dir.glob("*/callbacks/test_all_sentences.json")):
        folder = p.parent.parent.name
        try:
            data = json.loads(p.read_text())
            edits, total_chars = 0, 0
            for entry in data.values():
                true_s = entry["true"]
                pred_s = "".join(CHAR_INDEX.get(int(i), "?") for i in entry["pred"])
                edits += lev(pred_s, true_s)
                total_chars += len(true_s)
            cer = edits / max(total_chars, 1)
            rows.append((cer, folder, len(data)))
        except Exception as e:
            rows.append((999.0, f"{folder} (error: {e})", 0))

    for cer, folder, n in sorted(rows):
        print(f"{folder:<58} {cer:>10.3f} {n:>12}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
