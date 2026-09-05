#!/usr/bin/env python
"""Two-study runner for the SpanishBCBL 3-subject Brain2Qwerty cache.

Implements the two experiments described in `SpanishBCBL_3-Subj.txt`:

STUDY 1 — Single-subject benchmark (Section 2 of the guide)
    Train and test on subject S15 only (the recommended single-subject
    benchmark: high SNR, stable motor patterns), once with the Transformer
    sentence core (baseline) and once with the Mamba-2 core.

        train --subjects S15          --core {transformer,mamba} --small

STUDY 2 — Cross-subject transfer (Section 2.2 of the guide)
    Train jointly on S15 + S16, then evaluate zero-shot on the held-out
    subject S6. The model never sees S6 during training: the eval run
    rebuilds the data pipeline filtered to S6 and only runs the test split.

        train --subjects S15 S16      --core {transformer,mamba} --small
        eval  --subjects S6 --ckpt <best.ckpt> --core <same>     --small

Both studies are idempotent: a run whose `callbacks/test_all_sentences.json`
already exists is skipped (use --force to re-run), and an interrupted training
run is resumed from `last.ckpt` automatically.

After the runs finish, per-sentence predictions are scored with
`brain2qwerty_v1.scripts.extract_predictions` and a summary
(CSV + Markdown + JSON) is written to `two_studies_out/`.

Usage
-----
    # Preview the exact commands without running anything
    python run_two_studies.py --dry-run

    # Run both studies (4 training/eval jobs total)
    python run_two_studies.py

    # Only Study 2, Mamba core only
    python run_two_studies.py --study 2 --cores mamba

    # Just re-collect metrics from finished runs
    python run_two_studies.py --collect-only

Environment
-----------
Requires the same env vars as the training CLI:
    BRAIN2QWERTY_STUDIES  path to the extracted study (e.g. SpanishBCBL_3subj)
    BRAIN2QWERTY_CACHE    feature/exca cache dir (default ~/.cache/brain2qwerty)
Results land in $BRAIN2QWERTY_CACHE/results/small-<core>-<subjects>-<tag>/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "two_studies_out"

DEFAULT_CORES = ["transformer", "mamba"]  # baseline first, as in the guide
STUDY1_SUBJECTS = ["S15"]                   # single-subject benchmark
STUDY2_TRAIN_SUBJECTS = ["S15", "S16"]      # joint training pool
STUDY2_EVAL_SUBJECTS = ["S6"]               # held-out transfer target

SENTINEL = Path("callbacks") / "test_all_sentences.json"  # "run finished" marker


# --------------------------------------------------------------------------- #
# Run plan                                                                     #
# --------------------------------------------------------------------------- #

def results_root() -> Path:
    """Mirror of brain2qwerty_v1_mamba.config.xp_config.RESULTS."""
    if os.environ.get("BRAIN2QWERTY_RESULTS"):
        return Path(os.environ["BRAIN2QWERTY_RESULTS"])
    cache = os.environ.get("BRAIN2QWERTY_CACHE", str(Path.home() / ".cache" / "brain2qwerty"))
    return Path(cache) / "results"


def output_dir(core: str, subjects: list[str], tag: str, small: bool = True) -> Path:
    """Mirror of xp_config.experiment_config output_dir + CLI --tag suffix."""
    base = f"{'small-' if small else ''}{core}-" + "-".join(subjects)
    return results_root() / f"{base}-{tag}"


@dataclass
class Run:
    """One CLI invocation (a training run or a held-out evaluation)."""

    study: str                 # "study1" | "study2"
    kind: str                  # "train" | "eval"
    core: str
    subjects: list[str]
    tag: str
    label: str                 # human-readable arm name
    ckpt: Path | None = None   # eval only: checkpoint to load
    out: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out = output_dir(self.core, self.subjects, self.tag)

    @property
    def done(self) -> bool:
        return (self.out / SENTINEL).exists()


def build_plan(cores: list[str], studies: list[str],
               study1_subjects: list[str]) -> list[Run]:
    plan: list[Run] = []
    subj_slug = "-".join(s.lower() for s in study1_subjects)
    for core in cores:
        if "study1" in studies:
            plan.append(Run(
                study="study1", kind="train", core=core,
                subjects=study1_subjects, tag="study1",
                label=f"study1_{subj_slug}_{core}",
            ))
        if "study2" in studies:
            train = Run(
                study="study2", kind="train", core=core,
                subjects=STUDY2_TRAIN_SUBJECTS, tag="study2",
                label=f"study2_trainS15S16_{core}",
            )
            plan.append(train)
            plan.append(Run(
                study="study2", kind="eval", core=core,
                subjects=STUDY2_EVAL_SUBJECTS, tag="study2-evalS6",
                label=f"study2_evalS6_{core}",
                ckpt=train.out / "best.ckpt",
            ))
    return plan


def build_command(run: Run, python: str, preset: str, seed: int | None,
                  force: bool, require_ckpt: bool = True) -> list[str]:
    cmd = [python, "-m", "brain2qwerty_v1_mamba.main"]
    if run.kind == "train":
        # "colab" preset implies --small (T4-class single GPU); else plain train.
        cmd.append("colab" if preset == "colab" else "train")
        cmd += ["--subjects", *run.subjects, "--core", run.core]
        if preset != "colab":
            cmd.append("--small")
        if seed is not None:
            cmd += ["--seed", str(seed)]
        # Resume an interrupted run from last.ckpt (unless starting fresh).
        last_ckpt = run.out / "last.ckpt"
        if last_ckpt.exists() and not force:
            cmd += ["--resume", str(last_ckpt)]
    else:  # eval
        if run.ckpt is None or (require_ckpt and not run.ckpt.exists()):
            raise FileNotFoundError(
                f"checkpoint for {run.label} not found: {run.ckpt}\n"
                "Run the Study-2 training arm first."
            )
        cmd.append("eval")
        cmd += ["--subjects", *run.subjects, "--core", run.core,
                "--small", "--ckpt", str(run.ckpt)]
    cmd += ["--tag", run.tag]
    return cmd


# --------------------------------------------------------------------------- #
# Execution                                                                    #
# --------------------------------------------------------------------------- #

def check_environment() -> None:
    studies_path = Path(os.environ.get(
        "BRAIN2QWERTY_STUDIES", str(Path.home() / "brain2qwerty_data" / "studies")))
    if not studies_path.exists():
        print(f"WARNING: BRAIN2QWERTY_STUDIES path does not exist: {studies_path}")
        print("         Set it to the extracted study folder before real runs.\n")


def execute(plan: list[Run], python: str, preset: str, seed: int | None,
            dry_run: bool, force: bool) -> list[Run]:
    finished: list[Run] = []
    for run in plan:
        if run.done and not force:
            print(f"[skip] {run.label}: already finished "
                  f"({run.out / SENTINEL} exists)")
            finished.append(run)
            continue
        try:
            cmd = build_command(run, python, preset, seed, force,
                                require_ckpt=not dry_run)
        except FileNotFoundError as exc:
            print(f"[skip] {run.label}: {exc}")
            continue
        print(f"\n{'=' * 78}\n[{run.study}] {run.label}\n  out: {run.out}\n  cmd: "
              + " ".join(cmd) + f"\n{'=' * 78}")
        if dry_run:
            continue
        proc = subprocess.run(cmd, cwd=REPO)
        if proc.returncode != 0:
            print(f"[FAIL] {run.label}: exit code {proc.returncode} — "
                  "stopping here; fix the error and re-run (finished arms are skipped).")
            sys.exit(proc.returncode)
        if run.done:
            finished.append(run)
        else:
            print(f"[warn] {run.label}: process exited 0 but {SENTINEL} missing; "
                  "check the logs above.")
    return finished


# --------------------------------------------------------------------------- #
# Result collection                                                            #
# --------------------------------------------------------------------------- #

def _last_test_cer(run: Run) -> float | None:
    """Last non-null test_CER from the Lightning CSVLogger metrics file."""
    import csv

    metrics = run.out / "logs" / "version_0" / "metrics.csv"
    if not metrics.exists():
        return None
    value = None
    with open(metrics, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("test_CER")
            if raw not in (None, "", "nan"):
                value = float(raw)
    return value


def _score_predictions(run: Run, python: str) -> dict | None:
    """Per-subject + overall CER/WER from the predictions JSON."""
    json_path = run.out / SENTINEL
    if not json_path.exists():
        return None
    try:  # in-process (fast path)
        from brain2qwerty_v1.scripts.extract_predictions import process_json
        df = process_json(str(json_path))
    except Exception as exc:  # fallback: the script's own CLI
        print(f"  (in-process scoring unavailable: {exc}; using CLI fallback)")
        OUT_DIR.mkdir(exist_ok=True)
        csv_out = OUT_DIR / f"{run.label}_sentences.csv"
        proc = subprocess.run(
            [python, "-m", "brain2qwerty_v1.scripts.extract_predictions",
             "--input", str(run.out / "callbacks"), "--output", str(csv_out)],
            cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return None
        import pandas as pd
        df = pd.read_csv(csv_out)

    OUT_DIR.mkdir(exist_ok=True)
    df.drop(columns=["Logits"], errors="ignore").to_csv(
        OUT_DIR / f"{run.label}_sentences.csv", index=False)

    per_subject = df.groupby("Subject")[["CER", "WER"]].mean()
    n_subj = len(per_subject)
    sem = per_subject["CER"].std(ddof=1) / (n_subj ** 0.5) if n_subj > 1 else 0.0
    return {
        "n_sentences": int(len(df)),
        "CER_overall": float(per_subject["CER"].mean()),
        "CER_SEM": float(sem),
        "WER_overall": float(per_subject["WER"].mean()),
        "per_subject_CER": {s: float(r["CER"]) for s, r in per_subject.iterrows()},
    }


def collect(runs: list[Run], python: str) -> None:
    import csv as _csv

    rows, details = [], {}
    for run in runs:
        if not run.done:
            continue
        print(f"\n--- {run.label} ({run.out})")
        scores = _score_predictions(run, python)
        test_cer = _last_test_cer(run)
        row = {
            "study": run.study, "arm": run.label, "core": run.core,
            "subjects_in_split": "-".join(run.subjects), "kind": run.kind,
            "test_CER_metrics_csv": test_cer,
            "output_dir": str(run.out),
        }
        if scores:
            row.update({k: v for k, v in scores.items() if k != "per_subject_CER"})
            details[run.label] = scores
        rows.append(row)

    if not rows:
        print("\nNo finished runs to collect yet.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    summary_csv = OUT_DIR / "two_studies_summary.csv"
    cols = ["study", "arm", "core", "subjects_in_split", "kind",
            "n_sentences", "CER_overall", "CER_SEM", "WER_overall",
            "test_CER_metrics_csv", "output_dir"]
    with open(summary_csv, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "two_studies_summary.json").write_text(json.dumps(details, indent=2))

    lines = ["# Two-Study Results — SpanishBCBL 3-Subject Cache\n",
             "| Study | Arm | Core | Split subjects | CER (overall) | ± SEM | WER |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        cer = f"{r['CER_overall']:.1%}" if r.get("CER_overall") is not None else "—"
        sem = f"{r['CER_SEM']:.1%}" if r.get("CER_SEM") is not None else "—"
        wer = f"{r['WER_overall']:.1%}" if r.get("WER_overall") is not None else "—"
        lines.append(f"| {r['study']} | {r['arm']} | {r['core']} | "
                     f"{r['subjects_in_split']} | {cer} | {sem} | {wer} |")
    lines.append("\nStudy 2 arms with `kind=eval` are the zero-shot held-out-S6 "
                 "transfer evaluations (model trained on S15+S16 only).\n")
    (OUT_DIR / "two_studies_summary.md").write_text("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nSummary written to {summary_csv} (+ .json / .md next to it).")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["1", "2", "all"], default="all")
    ap.add_argument("--cores", nargs="+", default=DEFAULT_CORES,
                    choices=["transformer", "transformer_deep",
                             "mamba", "mamba3", "mamba_mlp", "mamba3_mlp",
                             "deltanet", "deltanet_mlp",
                             "hybrid", "hybrid3", "hybrid_8l", "hybrid3_8l"],
                    help="sentence cores to compare — every Stage-2 variant the "
                         "brain2qwerty_v1_mamba CLI implements (default: "
                         "transformer mamba). GNN / Conformer / "
                         "foundation-model encoders from the guide's ablation "
                         "table are NOT implemented in the codebase yet.")
    ap.add_argument("--study1-subjects", nargs="+", default=STUDY1_SUBJECTS,
                    help="subject(s) for the Study-1 benchmark (default: S15, "
                         "the guide's recommended single-subject benchmark; "
                         "e.g. --study1-subjects S16 or S15 S16 S6)")
    ap.add_argument("--preset", choices=["full", "colab"], default="full",
                    help="'colab' = single-GPU T4 preset (200 epochs, batch 32)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (default: 33)")
    ap.add_argument("--python", default=None,
                    help="python interpreter for the training CLI "
                         "(default: repo .venv if present, else this interpreter)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without executing them")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if results already exist (no resume)")
    ap.add_argument("--collect-only", action="store_true",
                    help="skip launching; only re-collect metrics from finished runs")
    args = ap.parse_args()

    studies = ["study1", "study2"] if args.study == "all" else [f"study{args.study}"]
    plan = build_plan(args.cores, studies, args.study1_subjects)

    venv_py = REPO / ".venv" / "bin" / "python"
    python = args.python or (str(venv_py) if venv_py.exists() else sys.executable)

    print(f"Interpreter : {python}")
    print(f"Results root: {results_root()}")
    print(f"Plan        : {len(plan)} run(s) — "
          + ", ".join(r.label for r in plan))

    if not args.collect_only:
        if not args.dry_run:
            check_environment()
        finished = execute(plan, python, args.preset, args.seed,
                           args.dry_run, args.force)
        if args.dry_run:
            return
    else:
        finished = [r for r in plan if r.done]

    collect(finished or [r for r in plan if r.done], python)


if __name__ == "__main__":
    main()
