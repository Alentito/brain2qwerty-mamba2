# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Event transforms for the V1-Mamba ablation.

Only one addition over V1: ``V1MambaSubjectFilter`` restricts the raw events
to a subset of participants. It runs BEFORE ``SpanishBCBLPreprocessing`` so
the participant merge rules (S18->S1, ...) and the integer factorisation
still happen inside V1's own transform, keeping the label space identical.
"""

import typing as tp

import pandas as pd
from neuralset.events.study import EventsTransform

from brain2qwerty_v1.utils import SUBJECT_MERGE

# Reverse of V1's merge map: canonical -> all raw ids recorded for that person.
_MERGED_IDS: dict[str, set[str]] = {}
for _raw, _canon in SUBJECT_MERGE.items():
    _MERGED_IDS.setdefault(_canon, set()).add(_raw)


def _normalise_subject(spec) -> str:
    """Accept 'S15', '15', 15 or 'Pinet2024Meg/S15' -> 'Pinet2024Meg/S15'."""
    s = str(spec).strip()
    if "/" in s:
        return s
    if not s.upper().startswith("S"):
        s = f"S{s}"
    return f"Pinet2024Meg/{s}"


class V1MambaSubjectFilter(EventsTransform):
    """Keep only the events of the requested participants.

    Subject specs are resolved to V1's canonical (post-merge) ids and expanded
    back to every raw recording id belonging to the same person, so selecting
    e.g. ``S1`` also keeps the duplicate recording ``S18``.
    """

    subjects: tp.Sequence[tp.Any] = ("S15", "S16", "S6")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        wanted_raw: set[str] = set()
        for spec in self.subjects:
            canon = _normalise_subject(spec)
            wanted_raw.add(canon)
            wanted_raw |= _MERGED_IDS.get(canon, set())
        out = events[events["subject"].isin(wanted_raw)]
        if out.empty:
            raise ValueError(
                f"V1MambaSubjectFilter: no events left for subjects "
                f"{sorted(wanted_raw)}; available: {sorted(events['subject'].unique())}"
            )
        return out.reset_index(drop=True)
