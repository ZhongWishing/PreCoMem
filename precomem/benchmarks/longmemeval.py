"""LongMemEval loader (Wu et al., 2024); we use the 500-question S variant.

Public layout (``longmemeval_s.json``)::

    [
        {
            "question_id": "...", "question": "...", "answer": "...",
            "question_type": "single-session-user",
            "haystack_sessions": [[{"role": "user", "content": "...", "has_answer": true}, ...], ...],
            "haystack_dates": ["2023/05/20 (Sat) 02:21", ...],
            "haystack_session_ids": ["...", ...],
            "question_date": "..."
        }, ...
    ]

Each record is one sample: its haystack is the dialogue, and its single
question is the QA pair. Question types map onto the six paper splits
(SSU / SSA / SSP / MS / KU / TR); the ``_abs`` suffix of abstention
questions is kept in ``QAPair.meta``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from .base import BenchmarkSample, DialogueTurn, QAPair, infer_role, load_json, unwrap_records

TYPE_MAP = {
    "single-session-user": "SSU",
    "single-session-assistant": "SSA",
    "single-session-preference": "SSP",
    "multi-session": "MS",
    "knowledge-update": "KU",
    "temporal-reasoning": "TR",
}


class LongMemEvalLoader:
    name = "longmemeval"

    @staticmethod
    def load(path: Union[str, Path]) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        for idx, item in enumerate(unwrap_records(load_json(path), "data")):
            qid = str(item.get("question_id") or item.get("id") or idx)
            sessions = item.get("haystack_sessions") or []
            dates = item.get("haystack_dates") or [None] * len(sessions)
            session_ids = item.get("haystack_session_ids") or [f"S{i + 1}" for i in range(len(sessions))]
            turns: List[DialogueTurn] = []
            for sid, date, session in zip(session_ids, dates, sessions):
                for entry in session:
                    speaker = str(entry.get("role") or entry.get("speaker") or "user")
                    turns.append(
                        DialogueTurn(
                            index=len(turns) + 1,
                            speaker=speaker,
                            text=str(entry.get("content") or entry.get("text") or ""),
                            role=infer_role(speaker),
                            session=str(sid),
                            timestamp=date,
                            meta={"has_answer": bool(entry.get("has_answer", False))},
                        )
                    )
            raw_type = str(item.get("question_type") or "unknown")
            abstention = qid.endswith("_abs")
            qa = QAPair(
                qid=qid,
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                category=TYPE_MAP.get(raw_type, raw_type),
                meta={"raw_type": raw_type, "abstention": abstention, "question_date": item.get("question_date")},
            )
            samples.append(
                BenchmarkSample(sample_id=qid, dialogue=turns, qa_pairs=[qa], meta={"source": "longmemeval"})
            )
        return samples
