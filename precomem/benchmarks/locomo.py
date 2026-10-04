"""LoCoMo loader (Maharana et al., 2024).

Expected layout is the public ``locomo10.json`` dump::

    {
        "sample_id": "...",
        "conversation": {
            "speaker_a": "Caroline", "speaker_b": "Melanie",
            "session_1": [{"speaker": "Caroline", "text": "...", "dia_id": "D1:0"}, ...],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            ...
        },
        "qa": [{"question": "...", "answer": "...", "category": 1, "evidence": ["D1:3"]}]
    }

Both participants are humans, so every message is treated as a user
observation attributed to its named speaker (see ``base.infer_role``).
Adversarial questions (category 5) are dropped unless ``include_adversarial``
is set, following the standard four-split protocol of the paper's Table 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from .base import BenchmarkSample, DialogueTurn, QAPair, load_json, unwrap_records

# Category numbering of the released LoCoMo data (snap-research/locomo):
#   1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop, 5 = adversarial.
# Table 1 reports the first four splits and excludes the adversarial one,
# which leaves 1,986 - 446 = 1,540 questions. Some third-party evaluation
# harnesses number the splits differently; pass ``category_names`` to switch.
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def _session_number(key: str) -> int:
    try:
        return int(key.split("_")[1])
    except (IndexError, ValueError):
        return 0


class LoCoMoLoader:
    name = "locomo"

    @staticmethod
    def load(
        path: Union[str, Path],
        category_names: Dict[int, str] = CATEGORY_NAMES,
        include_adversarial: bool = False,
    ) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        for idx, record in enumerate(unwrap_records(load_json(path), "data")):
            sample_id = str(record.get("sample_id") or record.get("id") or idx)
            conv: Dict[str, Any] = record.get("conversation") or record.get("dialogue") or {}
            session_keys = sorted(
                (k for k in conv if k.startswith("session_") and isinstance(conv[k], list)),
                key=_session_number,
            )
            turns: List[DialogueTurn] = []
            for key in session_keys:
                stamp = conv.get(f"{key}_date_time")
                for entry in conv[key]:
                    speaker = str(entry.get("speaker") or "user")
                    turns.append(
                        DialogueTurn(
                            index=len(turns) + 1,
                            speaker=speaker,
                            text=str(entry.get("text") or entry.get("content") or ""),
                            role="user",  # two humans: every message is an observation
                            session=key,
                            timestamp=stamp,
                            dialogue_id=entry.get("dia_id") or entry.get("dialog_id"),
                            meta={"blip_caption": entry.get("blip_caption")} if entry.get("blip_caption") else {},
                        )
                    )
            qa_pairs: List[QAPair] = []
            for j, q in enumerate(record.get("qa") or []):
                raw_cat = q.get("category")
                if not include_adversarial and category_names.get(raw_cat) == "adversarial":
                    continue
                qa_pairs.append(
                    QAPair(
                        qid=f"{sample_id}::Q{j:04d}",
                        question=str(q.get("question", "")),
                        answer=str(q.get("answer", q.get("adversarial_answer", ""))),
                        category=category_names.get(raw_cat, str(raw_cat)) if raw_cat is not None else None,
                        evidence=[str(e) for e in q.get("evidence") or []],
                        meta={"raw_category": raw_cat},
                    )
                )
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    dialogue=turns,
                    qa_pairs=qa_pairs,
                    meta={"source": "locomo", "speakers": [conv.get("speaker_a"), conv.get("speaker_b")]},
                )
            )
        return samples
