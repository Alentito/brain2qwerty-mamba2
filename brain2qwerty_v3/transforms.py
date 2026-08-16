# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import typing as tp

import numpy as np
import pandas as pd
from exca import MapInfra
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from neuralset.events.study import EventsTransform
from neuralset.extractors.text import BaseText

from brain2qwerty_v1.utils import select_participants

from .utils import key_to_int

logger = logging.getLogger(__name__)


class SpanishBCBLV2Preprocessing(EventsTransform):
    """Clean raw SpanishBCBL events and build the integer CTC target.

    First half mirrors V1's ``SpanishBCBLPreprocessing`` (practice-trial drop,
    participant selection, per-sentence ids, Sentence events rebuilt from
    keystrokes, ground-truth text propagation). Second half mirrors V2's
    ``EnglishBCBLPreprocessing`` (button normalisation, ``typed_key_int`` and
    the space-separated ``typed_label`` CTC target per sentence).

    RSVP perception-phase Word rows are dropped: the contrastive loss uses its
    own Word events created downstream by ``WordCreator``.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        # --- V1 cleaning -----------------------------------------------------
        events["type"] = events["type"].replace(
            {"Button": "Keystroke", "DetectedButton": "Keystroke"}
        )
        # drop the two practice trials of each block
        events = events[~events["trial_id"].isin([0.0, 1.0])]

        # keep the 19 unique participants (drop controls/excluded, merge duplicates)
        events = select_participants(events)
        events["subject"] = pd.factorize(events["subject"])[0]

        if "sentence_UID" not in events.columns:
            events["sentence_UID"] = (
                events["trial_id"].astype(str) + "_" + events["timeline"]
            )
        events = events[
            events["sentence_UID"] != "65.0_Pinet2024Meg_subject-S1_session-1_task-block1"
        ]

        # rebuild one Sentence event per keystroke group (spanning its keystrokes)
        events = events[events["type"] != "Sentence"]
        buttons = events[events["type"] == "Keystroke"]
        grouped = buttons.groupby("sentence_UID")
        updated_sentences = []
        for suid, df in grouped:
            sentence = df.iloc[0].copy()
            sentence["type"] = "Sentence"
            sentence["start"] = df["start"].min()
            sentence["stop"] = df["stop"].max()
            sentence["duration"] = df["stop"].max() - df["start"].min()
            updated_sentences.append(sentence)
        if updated_sentences:
            events = pd.concat([events, pd.DataFrame(updated_sentences)])
            events.reset_index(drop=True, inplace=True)

        # propagate the ground-truth sentence text to the Sentence events
        sentence_dict = buttons.set_index("sentence_UID")["sentence"].to_dict()
        events["text"] = events.apply(
            lambda row: (
                sentence_dict.get(row["sentence_UID"], None)
                if row["type"] == "Sentence"
                else row["text"]
            ),
            axis=1,
        )

        # --- V2 CTC target ----------------------------------------------------
        # normalise buttons: space -> "&", drop special/number tokens
        events = events[~events.button.isin(["<special>", "<number>"])]
        events.loc[events.button == "<space>", "button"] = "&"

        # drop keystrokes whose button is outside the CTC vocabulary
        unmapped = (events.type == "Keystroke") & ~events.button.isin(key_to_int)
        if unmapped.any():
            logger.info("Dropping %d keystroke(s) with unmapped buttons", unmapped.sum())
            events = events[~unmapped]

        # integer key id per keystroke
        button_events = events[events.type == "Keystroke"]
        events["typed_key_int"] = -1
        events["typed_key_int"] = events["typed_key_int"].astype(int)
        events.loc[button_events.index, "typed_key_int"] = button_events.button.map(
            key_to_int
        )

        # build the space-separated CTC target per sentence (skip near-empty ones)
        uids_to_drop: list[str] = []
        label_by_uid: dict[str, str] = {}
        for uid, group in tqdm(events.groupby("sentence_UID"), desc="Typed labels"):
            if "nan" in uid:
                continue
            ks = group[group.type == "Keystroke"].sort_values("start")
            if len(ks) == 0 or len(ks) < 0.5 * len(group):
                uids_to_drop.append(uid)
                continue
            typed_seq_ids = [int(i) for i in ks.typed_key_int.values]
            assert sum(i == -1 for i in typed_seq_ids) == 0, f"Unmapped keys in {uid}"
            label_by_uid[uid] = " ".join(str(i) for i in typed_seq_ids)
        events["typed_label"] = events["sentence_UID"].map(label_by_uid)
        if uids_to_drop:
            logger.info(
                "Dropping %d sentences with too few keystrokes", len(uids_to_drop)
            )
            events = events[~events.sentence_UID.isin(uids_to_drop)]

        # keep MEG + keystrokes + sentences that carry a label; drop the RSVP
        # perception-phase Word rows (WordCreator adds its own Word events)
        events = events[events.type.isin(["Sentence", "Keystroke", "Meg"])]
        is_sentence = events["type"] == "Sentence"
        has_label = events["typed_label"].notna() & (events["typed_label"] != "")
        events = events[~is_sentence | has_label]

        # stable per-keystroke id (ordered within each sentence)
        keystroke_mask = events["type"] == "Keystroke"
        ks = events.loc[keystroke_mask].sort_values("start")
        if len(ks) > 0:
            counter = ks.groupby("sentence_UID").cumcount() + 1
            events.loc[keystroke_mask, "button_UID"] = (
                ks["sentence_UID"] + "_button_" + counter.astype(str)
            )
        return events


class SpanishBCBLV2Splitter(EventsTransform):
    """V1's TF-IDF paraphrase-cluster split, propagated to every row.

    Sentences are clustered by TF-IDF cosine similarity and clusters allocated
    greedily to the splits until the target ratios (in keystrokes) are met —
    identical logic and seed to the V1 reproduction, so the test sentences match
    the reported baseline exactly. Unlike V1 (which tagged only keystroke rows),
    the split is assigned per ``sentence_UID`` so Sentence rows carry it too, as
    required by the V2 sentence-level dataloader.
    """

    splitting_ratios: tuple = (0.8, 0.1, 0.1)
    seed: int = 1
    threshold: float = 0.5

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        buttons = events[events["type"] == "Keystroke"]
        unique_sentences = buttons["sentence"].unique()
        random.seed(self.seed)

        # cluster paraphrase-similar sentences via TF-IDF cosine similarity
        tfidf = TfidfVectorizer().fit_transform(unique_sentences)
        sim = cosine_similarity(tfidf)
        clusters: list[list[int]] = []
        visited: set[int] = set()
        for i in range(sim.shape[0]):
            if i in visited:
                continue
            cluster = {i}
            expanded = True
            while expanded:
                expanded = False
                for idx in list(cluster):
                    for j in range(sim.shape[1]):
                        if j not in cluster and sim[idx, j] > self.threshold:
                            cluster.add(j)
                            expanded = True
            visited.update(cluster)
            clusters.append(list(cluster))
        random.shuffle(clusters)

        # allocate clusters to splits to hit the target keystroke ratios
        total = len(buttons)
        sizes = {
            "train": int(self.splitting_ratios[0] * total),
            "val": int(self.splitting_ratios[1] * total),
            "test": total
            - int(self.splitting_ratios[0] * total)
            - int(self.splitting_ratios[1] * total),
        }
        current = {"train": 0, "val": 0, "test": 0}
        sentence_to_split: dict[str, str] = {}
        for cluster in clusters:
            cluster_sents = [unique_sentences[idx] for idx in cluster]
            cluster_size = len(buttons[buttons["sentence"].isin(cluster_sents)])
            assigned = "test"
            for split in ["train", "val", "test"]:
                if current[split] + cluster_size <= sizes[split]:
                    current[split] += cluster_size
                    assigned = split
                    break
            for s in cluster_sents:
                sentence_to_split[s] = assigned

        # propagate to every row via sentence_UID (Sentence rows carry "text")
        text_by_uid = {
            row["sentence_UID"]: row["text"]
            for _, row in events[events["type"] == "Sentence"].iterrows()
        }
        uid_to_split = {
            uid: sentence_to_split.get(text)
            for uid, text in text_by_uid.items()
        }
        events["split"] = events["sentence_UID"].map(uid_to_split)
        return events


class WordCreator(EventsTransform):
    """Create one Word event per whitespace token of each Sentence.

    Each Word inherits the parent Sentence's identifiers/timing and records its
    ``word_order`` and left ``context`` so contextualised text embeddings can be
    computed as the contrastive target.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        sents = events[events["type"] == "Sentence"]
        if sents.empty:
            return events
        inherit_cols = [
            c
            for c in ("timeline", "subject", "start", "duration", "sentence_UID")
            if c in events.columns
        ]
        word_rows: list[dict] = []
        for _, row in sents.iterrows():
            words = str(row["text"]).strip().split()
            for idx, word in enumerate(words):
                wr = {
                    "type": "Word",
                    "text": word,
                    "sentence": str(row["text"]).strip(),
                    "context": " ".join(words[: idx + 1]),
                    "word_order": idx,
                }
                wr.update({col: row[col] for col in inherit_cols})
                word_rows.append(wr)
        if not word_rows:
            return events
        return pd.concat([events, pd.DataFrame(word_rows)], ignore_index=True)


class SentenceKeySeq(BaseText):
    """Turn each sentence into the integer character sequence the CTC head predicts.

    Two ways to build that sequence:
    - ``mode="typed_label"`` uses what the participant actually typed (the integer
      sequence precomputed per sentence in ``event.extra["typed_label"]``).
    - ``mode="sentence_text"`` uses the reference sentence text: lowercase it, map
      spaces to ``&`` and each character to its index via ``key_to_int``.
    """

    event_types: str | tuple[str, ...] = "Sentence"
    mode: tp.Literal["typed_label", "sentence_text"] = "typed_label"

    infra: MapInfra = MapInfra(version="v5")

    @infra.apply(
        item_uid=lambda event: str(event.text),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: list[tp.Any]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            events = tqdm(events, desc="Sequence labels")  # type: ignore
        for event in events:
            yield self.get_embedding(event)

    def get_embedding(self, event) -> np.ndarray:
        if self.mode == "typed_label":
            return np.array(
                [int(i) for i in event.extra["typed_label"].split(" ")], dtype=np.int32
            )
        text = str(event.text).lower().replace(" ", "&")
        seq = [key_to_int[ch] for ch in text if ch in key_to_int]
        if not seq:
            raise ValueError(f"Empty target for text={event.text!r}")
        return np.array(seq, dtype=np.int32)
