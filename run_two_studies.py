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

    # STUDY 4 (v3 pipeline): encoder x core grid on the asynchronous pipeline
    python run_two_studies.py --pipeline v3 --study 1 --study1-subjects S16 \
        --encoders conv gnn --cores conformer mamba3_hybrid_stabilized deltanet \
        --dry-run

Pipelines
---------
--pipeline v1 (default; behavior unchanged)
    brain2qwerty_v1_mamba synchronous keystroke decoding. Sentinel for a
    finished run: `callbacks/test_all_sentences.json`. Collected: per-sentence
    CER/WER/sentence-accuracy via brain2qwerty_v1.scripts.extract_predictions
    plus the last test_CER from the Lightning metrics.csv.

--pipeline v3 (Study 4)
    brain2qwerty_v3 asynchronous word-level decoding. Arms:
        python -m brain2qwerty_v3.main train --subjects <S...> \
            --core {conformer,mamba3_hybrid_stabilized,deltanet,...} \
            --frontend {conv,gnn} --tag study4
    Output dirs mirror v3's convention:
        $RESULTS/v3-[gnn-]<core>-study4/
    Sentinel for a finished run: `training_profile.json` — written
    unconditionally by TrainingTimeProfilingCallback.on_test_end (rank 0),
    so it exists even for runs stopped before the LLM stage (unlike
    `predictions_test.csv`, which is only written when per-sentence LLM
    predictions exist).
    Collected per finished arm (whatever exists — v3 does not dump v1-style
    per-sentence JSON, so metrics.csv scraping is the primary source):
      * last `test/cer_epo` (CTC CER), `test/CER`, `test/WER`, `test/SemER`
        from logs/version_0/metrics.csv
      * mean CER / WER / CTC_CER / SemER over `predictions_test.csv` when the
        LLM stage produced per-sentence predictions.

Environment
-----------
Requires the same env vars as the training CLI:
    BRAIN2QWERTY_STUDIES  path to the extracted study (e.g. SpanishBCBL_3subj)
    BRAIN2QWERTY_CACHE    feature/exca cache dir (default ~/.cache/brain2qwerty
                          for v1, ~/.cache/b2q_v1mamba for v3 — mirrors each
                          pipeline's own xp_config)
v1 results land in $BRAIN2QWERTY_CACHE/results/small-[gnn-]<core>-<subjects>-<tag>/
(the `gnn-` infix appears only for --encoders gnn arms).
v3 results land in <v3 RESULTS>/v3-[gnn-]<core>-study4/.
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
V1_CORES = ["transformer", "transformer_deep",
            "mamba", "mamba3", "mamba_mlp", "mamba3_mlp",
            "deltanet", "deltanet_mlp",
            "hybrid", "hybrid3", "hybrid_8l", "hybrid3_8l"]
# Study 4 (v3): every sequence core the brain2qwerty_v3 CLI implements;
# the default grid is the Study-4 comparison set.
V3_CORES = ["conformer", "mamba_mlp", "mamba3_hybrid_stabilized", "hybrid", "deltanet"]
DEFAULT_CORES_V3 = ["conformer", "mamba3_hybrid_stabilized", "deltanet"]
STUDY1_SUBJECTS = ["S15"]                   # single-subject benchmark
STUDY2_TRAIN_SUBJECTS = ["S15", "S16"]      # joint training pool
STUDY2_EVAL_SUBJECTS = ["S6"]               # held-out transfer target

SENTINEL = Path("callbacks") / "test_all_sentences.json"  # v1 "run finished" marker
# v3 "run finished" marker: written unconditionally at test end (rank 0) by
# TrainingTimeProfilingCallback — see the module docstring for why not
# predictions_test.csv.
SENTINEL_V3 = Path("training_profile.json")
V3_TAG = "study4"


# --------------------------------------------------------------------------- #
# Run plan                                                                     #
# --------------------------------------------------------------------------- #

def results_root() -> Path:
    """Mirror of brain2qwerty_v1_mamba.config.xp_config.RESULTS."""
    if os.environ.get("BRAIN2QWERTY_RESULTS"):
        return Path(os.environ["BRAIN2QWERTY_RESULTS"])
    cache = os.environ.get("BRAIN2QWERTY_CACHE", str(Path.home() / ".cache" / "brain2qwerty"))
    return Path(cache) / "results"


def results_root_v3() -> Path:
    """Mirror of brain2qwerty_v3.config.xp_config RESULTS (incl. its cache
    default ``~/.cache/b2q_v1mamba`` and the cluster sharedscratch path)."""
    if os.environ.get("BRAIN2QWERTY_RESULTS"):
        return Path(os.environ["BRAIN2QWERTY_RESULTS"])
    if os.environ.get("BRAIN2QWERTY_CACHE"):
        cache = Path(os.environ["BRAIN2QWERTY_CACHE"])
    else:
        cluster_cache = Path.home() / "sharedscratch" / "B2Q" / "cache_v1mamba"
        cache = cluster_cache if cluster_cache.exists() else (
            Path.home() / ".cache" / "b2q_v1mamba")
    return cache / "results"


def output_dir(core: str, subjects: list[str], tag: str, small: bool = True,
               encoder: str = "conv") -> Path:
    """Mirror of xp_config.experiment_config output_dir + CLI --tag suffix."""
    base = (f"{'small-' if small else ''}{'gnn-' if encoder == 'gnn' else ''}{core}-"
            + "-".join(subjects))
    return results_root() / f"{base}-{tag}"


def output_dir_v3(core: str, encoder: str, tag: str) -> Path:
    """Mirror of brain2qwerty_v3.main.tagged_output_dir:
    ``v3-[gnn-]<core>-<tag>`` (conv naming exactly as before Study 4)."""
    return results_root_v3() / f"v3-{'gnn-' if encoder == 'gnn' else ''}{core}-{tag}"


@dataclass
class Run:
    """One CLI invocation (a training run or a held-out evaluation)."""

    study: str                 # "study1" | "study2" | "study4"
    kind: str                  # "train" | "eval"
    core: str
    subjects: list[str]
    tag: str
    label: str                 # human-readable arm name
    encoder: str = "conv"      # "conv" | "gnn" (Stage-1 window encoder / frontend)
    pipeline: str = "v1"       # "v1" (brain2qwerty_v1_mamba) | "v3" (brain2qwerty_v3)
    ckpt: Path | None = None   # eval only: checkpoint to load
    out: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.pipeline == "v3":
            self.out = output_dir_v3(self.core, self.encoder, self.tag)
        else:
            self.out = output_dir(self.core, self.subjects, self.tag,
                                  encoder=self.encoder)

    @property
    def sentinel(self) -> Path:
        return SENTINEL_V3 if self.pipeline == "v3" else SENTINEL

    @property
    def done(self) -> bool:
        return (self.out / self.sentinel).exists()


def build_plan(encoders: list[str], cores: list[str], studies: list[str],
               study1_subjects: list[str]) -> list[Run]:
    plan: list[Run] = []
    subj_slug = "-".join(s.lower() for s in study1_subjects)
    for encoder in encoders:
        # conv arms keep the historical label/dir naming exactly
        core_slug = lambda c: f"gnn-{c}" if encoder == "gnn" else c  # noqa: E731
        for core in cores:
            if "study1" in studies:
                plan.append(Run(
                    study="study1", kind="train", core=core, encoder=encoder,
                    subjects=study1_subjects, tag="study1",
                    label=f"study1_{subj_slug}_{core_slug(core)}",
                ))
            if "study2" in studies:
                train = Run(
                    study="study2", kind="train", core=core, encoder=encoder,
                    subjects=STUDY2_TRAIN_SUBJECTS, tag="study2",
                    label=f"study2_trainS15S16_{core_slug(core)}",
                )
                plan.append(train)
                plan.append(Run(
                    study="study2", kind="eval", core=core, encoder=encoder,
                    subjects=STUDY2_EVAL_SUBJECTS, tag="study2-evalS6",
                    label=f"study2_evalS6_{core_slug(core)}",
                    ckpt=train.out / "best.ckpt",
                ))
    return plan


def build_plan_v3(encoders: list[str], cores: list[str],
                  subjects: list[str]) -> list[Run]:
    """Study 4 (v3): the encoder x core cartesian product, one training arm
    per combination on the given subject(s), tagged ``study4``."""
    plan: list[Run] = []
    subj_slug = "-".join(s.lower() for s in subjects)
    for encoder in encoders:
        core_slug = lambda c: f"gnn-{c}" if encoder == "gnn" else c  # noqa: E731
        for core in cores:
            plan.append(Run(
                study="study4", kind="train", core=core, encoder=encoder,
                pipeline="v3", subjects=subjects, tag=V3_TAG,
                label=f"study4_{subj_slug}_{core_slug(core)}",
            ))
    return plan


def build_command(run: Run, python: str, preset: str, seed: int | None,
                  force: bool, require_ckpt: bool = True) -> list[str]:
    if run.pipeline == "v3":
        return build_command_v3(run, python, seed, force)
    cmd = [python, "-m", "brain2qwerty_v1_mamba.main"]
    if run.kind == "train":
        # "colab" preset implies --small (T4-class single GPU); else plain train.
        cmd.append("colab" if preset == "colab" else "train")
        cmd += ["--subjects", *run.subjects, "--core", run.core,
                "--encoder", run.encoder]
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
                "--encoder", run.encoder, "--small", "--ckpt", str(run.ckpt)]
    cmd += ["--tag", run.tag]
    return cmd


def build_command_v3(run: Run, python: str, seed: int | None,
                     force: bool) -> list[str]:
    """v3 (Study 4) arm:
    ``python -m brain2qwerty_v3.main train --subjects ... --core ... --frontend ... --tag study4``.
    """
    cmd = [python, "-m", "brain2qwerty_v3.main", "train",
           "--subjects", *run.subjects,
           "--core", run.core, "--frontend", run.encoder,
           "--tag", run.tag]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    # Resume an interrupted run from last.ckpt (save_last=True), unless fresh.
    last_ckpt = run.out / "last.ckpt"
    if last_ckpt.exists() and not force:
        cmd += ["--resume", str(last_ckpt)]
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
                  f"({run.out / run.sentinel} exists)")
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
            print(f"[warn] {run.label}: process exited 0 but {run.sentinel} missing; "
                  "check the logs above.")
    return finished


# --------------------------------------------------------------------------- #
# Result collection                                                            #
# --------------------------------------------------------------------------- #

def _last_metric(run: Run, column: str) -> float | None:
    """Last non-null value of ``column`` from the Lightning CSVLogger metrics file."""
    import csv

    metrics = run.out / "logs" / "version_0" / "metrics.csv"
    if not metrics.exists():
        return None
    value = None
    with open(metrics, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get(column)
            if raw not in (None, "", "nan"):
                value = float(raw)
    return value


def _last_test_cer(run: Run) -> float | None:
    """Last non-null test_CER from the Lightning CSVLogger metrics file."""
    return _last_metric(run, "test_CER")


def _collect_v3(run: Run) -> dict:
    """Scrape whatever a finished v3 (Study 4) run wrote.

    Primary source: logs/version_0/metrics.csv —
      * ``test/cer_epo``: test CTC CER (always present; the stage-1 metric)
      * ``test/CER`` / ``test/WER`` / ``test/SemER``: LLM-decoder metrics
        (present once the LLM stage ran at test time)
    Secondary: predictions_test.csv (per-sentence LLM predictions, only when
    the LLM stage produced rows) -> mean CER / WER / CTC_CER / SemER.
    """
    import csv

    row = {
        "test_CTC_CER": _last_metric(run, "test/cer_epo"),
        "test_LLM_CER": _last_metric(run, "test/CER"),
        "test_LLM_WER": _last_metric(run, "test/WER"),
        "test_LLM_SemER": _last_metric(run, "test/SemER"),
    }
    preds = run.out / "predictions_test.csv"
    if preds.exists():
        with open(preds, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            row["n_sentences"] = len(rows)
            for src, dst in (("CER", "pred_CER_mean"), ("WER", "pred_WER_mean"),
                             ("CTC_CER", "pred_CTC_CER_mean"),
                             ("SemER", "pred_SemER_mean")):
                vals = [float(r[src]) for r in rows if r.get(src) not in (None, "")]
                if vals:
                    row[dst] = sum(vals) / len(vals)
    return row


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
    # sentence accuracy: fraction of sentences decoded with zero CER
    # (exact whole-sentence match), averaged per subject like CER/WER
    sent_acc = df.assign(exact=(df["CER"] == 0).astype(float)).groupby(
        "Subject")["exact"].mean()
    n_subj = len(per_subject)
    sem = per_subject["CER"].std(ddof=1) / (n_subj ** 0.5) if n_subj > 1 else 0.0
    return {
        "n_sentences": int(len(df)),
        "CER_overall": float(per_subject["CER"].mean()),
        "CER_SEM": float(sem),
        "WER_overall": float(per_subject["WER"].mean()),
        "sentence_accuracy": float(sent_acc.mean()),
        "per_subject_CER": {s: float(r["CER"]) for s, r in per_subject.iterrows()},
    }


def collect(runs: list[Run], python: str) -> None:
    import csv as _csv

    rows, details = [], {}
    for run in runs:
        if not run.done:
            continue
        print(f"\n--- {run.label} ({run.out})")
        row = {
            "study": run.study, "arm": run.label, "encoder": run.encoder,
            "core": run.core, "pipeline": run.pipeline,
            "subjects_in_split": "-".join(run.subjects), "kind": run.kind,
            "output_dir": str(run.out),
        }
        if run.pipeline == "v3":
            # v3 writes no v1-style per-sentence JSON; scrape metrics.csv
            # (+ predictions_test.csv means when the LLM stage produced rows).
            row.update(_collect_v3(run))
        else:
            row["test_CER_metrics_csv"] = _last_test_cer(run)
            scores = _score_predictions(run, python)
            if scores:
                row.update({k: v for k, v in scores.items() if k != "per_subject_CER"})
                details[run.label] = scores
        rows.append(row)

    if not rows:
        print("\nNo finished runs to collect yet.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    summary_csv = OUT_DIR / "two_studies_summary.csv"
    cols = ["study", "arm", "pipeline", "encoder", "core", "subjects_in_split",
            "kind", "n_sentences", "CER_overall", "CER_SEM", "WER_overall",
            "sentence_accuracy", "test_CER_metrics_csv",
            "test_CTC_CER", "test_LLM_CER", "test_LLM_WER", "test_LLM_SemER",
            "pred_CER_mean", "pred_WER_mean", "pred_CTC_CER_mean",
            "pred_SemER_mean", "output_dir"]
    with open(summary_csv, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "two_studies_summary.json").write_text(json.dumps(details, indent=2))

    v1_rows = [r for r in rows if r.get("pipeline") == "v1"]
    v3_rows = [r for r in rows if r.get("pipeline") == "v3"]

    lines = ["# Two-Study Results — SpanishBCBL 3-Subject Cache\n"]
    if v1_rows:
        lines += ["## V1 pipeline (brain2qwerty_v1_mamba)\n",
                  "| Study | Arm | Encoder | Core | Split subjects | CER (overall) | ± SEM | WER | Sentence acc |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in v1_rows:
            cer = f"{r['CER_overall']:.1%}" if r.get("CER_overall") is not None else "—"
            sem = f"{r['CER_SEM']:.1%}" if r.get("CER_SEM") is not None else "—"
            wer = f"{r['WER_overall']:.1%}" if r.get("WER_overall") is not None else "—"
            sacc = (f"{r['sentence_accuracy']:.1%}"
                    if r.get("sentence_accuracy") is not None else "—")
            lines.append(f"| {r['study']} | {r['arm']} | {r['encoder']} | {r['core']} | "
                         f"{r['subjects_in_split']} | {cer} | {sem} | {wer} | {sacc} |")
        lines.append("\nStudy 2 arms with `kind=eval` are the zero-shot held-out-S6 "
                     "transfer evaluations (model trained on S15+S16 only).\n")
    if v3_rows:
        def _pct(r, k):
            return f"{r[k]:.1%}" if r.get(k) is not None else "—"
        lines += ["## V3 pipeline (brain2qwerty_v3, Study 4)\n",
                  "| Arm | Frontend | Core | Test CTC CER | Test LLM CER | Test LLM WER | Pred CTC CER (mean) | Pred WER (mean) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in v3_rows:
            lines.append(
                f"| {r['arm']} | {r['encoder']} | {r['core']} | "
                f"{_pct(r, 'test_CTC_CER')} | {_pct(r, 'test_LLM_CER')} | "
                f"{_pct(r, 'test_LLM_WER')} | {_pct(r, 'pred_CTC_CER_mean')} | "
                f"{_pct(r, 'pred_WER_mean')} |")
        lines.append("\nV3 metrics are scraped from `logs/version_0/metrics.csv` "
                     "(last test value) and `predictions_test.csv` means; v3 does "
                     "not dump v1-style per-sentence JSON.\n")
    (OUT_DIR / "two_studies_summary.md").write_text("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nSummary written to {summary_csv} (+ .json / .md next to it).")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["v1", "v3"], default="v1",
                    help="v1 = brain2qwerty_v1_mamba Studies 1+2 (default, "
                         "unchanged); v3 = brain2qwerty_v3 Study-4 encoder x "
                         "core grid (--tag study4).")
    ap.add_argument("--study", choices=["1", "2", "all"], default="all")
    ap.add_argument("--cores", nargs="+", default=None,
                    help="cores to compare. v1 accepts: " + " ".join(V1_CORES) +
                         " (default: transformer mamba). v3 accepts: " +
                         " ".join(V3_CORES) +
                         " (default: conformer mamba3_hybrid_stabilized deltanet).")
    ap.add_argument("--encoders", nargs="+", default=["conv"],
                    choices=["conv", "gnn"],
                    help="Stage-1 encoders/frontends to compare (default: conv). "
                         "The run plan is the cartesian product encoder x core; "
                         "gnn arms get a gnn- label/output-dir infix.")
    ap.add_argument("--study1-subjects", nargs="+", default=STUDY1_SUBJECTS,
                    help="subject(s) for the Study-1 benchmark (default: S15, "
                         "the guide's recommended single-subject benchmark; "
                         "e.g. --study1-subjects S16 or S15 S16 S6). Also the "
                         "subject(s) of the v3 Study-4 grid.")
    ap.add_argument("--preset", choices=["full", "colab"], default="full",
                    help="'colab' = single-GPU T4 preset (200 epochs, batch 32); "
                         "v1 only")
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

    if args.pipeline == "v3":
        if args.study == "2":
            print("NOTE: Study 4 on the v3 pipeline defines only the "
                  "single-subject encoder x core grid; ignoring --study 2.")
        cores = args.cores or DEFAULT_CORES_V3
        bad = [c for c in cores if c not in V3_CORES]
        if bad:
            ap.error(f"--pipeline v3 cores must be among {V3_CORES}; got {bad}")
        plan = build_plan_v3(args.encoders, cores, args.study1_subjects)
        root = results_root_v3()
    else:
        cores = args.cores or DEFAULT_CORES
        bad = [c for c in cores if c not in V1_CORES]
        if bad:
            ap.error(f"--pipeline v1 cores must be among {V1_CORES}; got {bad}")
        studies = ["study1", "study2"] if args.study == "all" else [f"study{args.study}"]
        plan = build_plan(args.encoders, cores, studies, args.study1_subjects)
        root = results_root()

    venv_py = REPO / ".venv" / "bin" / "python"
    python = args.python or (str(venv_py) if venv_py.exists() else sys.executable)

    print(f"Interpreter : {python}")
    print(f"Results root: {root}")
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
