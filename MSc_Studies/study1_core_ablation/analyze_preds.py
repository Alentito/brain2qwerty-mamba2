"""Per-subject CER analysis for the V1-Mamba ablation.

Reads the two test_all_sentences.json prediction files (one per arm),
decodes char indices via brain2qwerty_v1.utils.CHAR_INDEX, and computes
sentence-level CER (Levenshtein / len(typed)) per subject and pooled.

Usage: python analyze_preds.py
"""

import ast
import json
import re
from pathlib import Path

# CHAR_INDEX lives in brain2qwerty_v1/utils.py, which imports torch at module
# level — parse the dict literal statically instead of importing it.
_src = (Path(__file__).parent / "brain2qwerty_v1" / "utils.py").read_text()
_node = ast.parse(_src)
CHAR_INDEX = None
for n in ast.walk(_node):
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CHAR_INDEX":
        CHAR_INDEX = ast.literal_eval(n.value)
        break
assert CHAR_INDEX, "CHAR_INDEX not found in brain2qwerty_v1/utils.py"

HERE = Path(__file__).parent
ARMS = {
    "mamba": HERE / "preds_mamba.json",
    "transformer": HERE / "preds_transformer.json",
}


def lev(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def decode(indices) -> str:
    return "".join(CHAR_INDEX.get(int(i), "?") for i in indices)


def cer_by_subject(path: Path):
    data = json.loads(path.read_text())
    per_subj: dict[str, list[tuple[int, int]]] = {}
    n = 0
    for key, rec in data.items():
        m = re.search(r"(S\d+)", key)
        if m is None:
            continue
        subj = m.group(1)
        pred = decode(rec["pred"])
        typed = decode(rec["typed"])
        d = lev(pred, typed)
        per_subj.setdefault(subj, []).append((d, len(typed)))
        n += 1
    out = {}
    for subj, pairs in sorted(per_subj.items()):
        edits = sum(p[0] for p in pairs)
        chars = sum(p[1] for p in pairs)
        out[subj] = {"cer": edits / chars, "sentences": len(pairs), "chars": chars}
    pooled_e = sum(p[0] for ps in per_subj.values() for p in ps)
    pooled_c = sum(p[1] for ps in per_subj.values() for p in ps)
    out["POOLED"] = {"cer": pooled_e / pooled_c,
                     "sentences": n,
                     "chars": pooled_c}
    return out


def main():
    results = {arm: cer_by_subject(p) for arm, p in ARMS.items()}
    subjects = sorted({s for r in results.values() for s in r if s != "POOLED"})

    print(f"\n{'subject':<10}{'mamba CER':>12}{'transformer CER':>17}{'n_sent':>8}")
    print("-" * 47)
    for s in subjects + ["POOLED"]:
        m, t = results["mamba"].get(s), results["transformer"].get(s)
        nsent = (m or t)["sentences"]
        print(f"{s:<10}{m['cer']:>12.3f}{t['cer']:>17.3f}{nsent:>8}")

    # cross-check vs Lightning test_CER printed in the slurm logs
    print("\nLightning test_CER from logs: mamba 0.412, transformer 0.285")


if __name__ == "__main__":
    main()
