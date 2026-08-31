# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from pathlib import Path

import lightning.pytorch as pl

from .utils import compute_sample_metrics, prediction_fieldnames


class PredictionCSVCallback(pl.Callback):
    """Save per-sentence predictions (text + CER/WER/SemER) to ``predictions_test.csv``.

    Only written at test time (not every validation epoch). The rows accumulated by
    the Lightning module are gathered across ranks first, which is a no-op for the
    single-process ``eval`` and keeps the training-end test complete on multi-GPU.
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)

    @staticmethod
    def _gather_rows(trainer, rows: list[dict]) -> list[dict]:
        if trainer.world_size <= 1:
            return rows
        import torch.distributed as dist

        gathered: list[list[dict] | None] = [None] * trainer.world_size
        dist.all_gather_object(gathered, rows)
        if trainer.global_rank == 0:
            return [r for rank_rows in gathered for r in rank_rows]
        return []

    def _save(self, trainer, rows, filename, with_semer):
        rows = self._gather_rows(trainer, rows)
        if not rows or trainer.global_rank != 0:
            return
        ctc_texts = [r.get("ctc_text", "") for r in rows]
        has_ctc = any(ctc_texts)
        has_segment_meta = any(r.get("subject") for r in rows)
        rows_with_metrics = compute_sample_metrics(
            [r["true_text"] for r in rows],
            [r["pred_text"] for r in rows],
            ctc_texts=ctc_texts if has_ctc else None,
            with_semer=with_semer,
        )
        for row_m, row_raw in zip(rows_with_metrics, rows):
            if has_segment_meta:
                row_m["subject"] = row_raw.get("subject", "")
                row_m["sentence_UID"] = row_raw.get("sentence_UID", "")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=prediction_fieldnames(has_ctc, has_segment_meta)
            )
            writer.writeheader()
            writer.writerows(rows_with_metrics)
        print(f"Saved {len(rows_with_metrics)} predictions to {path}")

    def on_test_epoch_end(self, trainer, pl_module):
        rows = getattr(pl_module, "_test_predictions", [])
        self._save(trainer, rows, "predictions_test.csv", with_semer=True)


class TrainingTimeProfilingCallback(pl.Callback):
    """Profile and save training time, epoch durations, and inference latency."""

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.fit_start_time: float = 0.0
        self.test_start_time: float = 0.0
        self.epoch_start_time: float = 0.0
        self.epoch_times: list[float] = []

    def on_fit_start(self, trainer, pl_module):
        import time
        self.fit_start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        import time
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)

    def on_test_start(self, trainer, pl_module):
        import time
        self.test_start_time = time.time()

    def on_test_end(self, trainer, pl_module):
        import time
        import json

        if trainer.global_rank != 0:
            return

        total_train_sec = (time.time() - self.fit_start_time) if self.fit_start_time > 0 else 0.0
        total_test_sec = time.time() - self.test_start_time
        n_epochs = len(self.epoch_times)
        mean_epoch_sec = float(sum(self.epoch_times) / max(1, n_epochs))
        
        n_test_samples = len(getattr(pl_module, "_test_predictions", []))
        ms_per_sentence = (total_test_sec / max(1, n_test_samples)) * 1000.0

        profile_data = {
            "total_training_time_seconds": round(total_train_sec, 2),
            "total_training_time_hours": round(total_train_sec / 3600.0, 3),
            "total_epochs_trained": n_epochs,
            "mean_epoch_duration_seconds": round(mean_epoch_sec, 2),
            "total_test_inference_time_seconds": round(total_test_sec, 2),
            "test_sample_count": n_test_samples,
            "inference_latency_per_sentence_ms": round(ms_per_sentence, 2),
            "epoch_durations_seconds": [round(t, 2) for t in self.epoch_times],
        }

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / "training_profile.json"
        with open(out_path, "w") as f:
            json.dump(profile_data, f, indent=2)

        print("\n" + "=" * 80)
        print("⏱️  TRAINING & INFERENCE TIMING PROFILE")
        print("=" * 80)
        print(f"Total Training Time:           {profile_data['total_training_time_hours']:.2f} hours ({profile_data['total_training_time_seconds']:.1f} s)")
        print(f"Total Epochs Trained:          {n_epochs}")
        print(f"Mean Epoch Duration:           {profile_data['mean_epoch_duration_seconds']:.2f} s")
        print(f"Test Set Inference Time:       {profile_data['total_test_inference_time_seconds']:.2f} s")
        print(f"Inference Latency/Sentence:    {profile_data['inference_latency_per_sentence_ms']:.2f} ms")
        print(f"Profile saved to:              {out_path}")
        print("=" * 80 + "\n")
