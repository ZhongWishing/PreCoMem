"""PersonaMem-v2 loader (Jiang et al., 2025); 32k and 128k variants share one layout::

    {
        "persona_id": "...",
        "history": [{"role": "user", "content": "...", "session": 1}, ...],
        "questions": [
            {"question_id": "...", "question": "...", "options": ["A. ...", ...],
             "answer": "B", "category": "anti_stereotype", "format": "MCQ"}
        ]
    }

Open-ended questions are accepted as well (``options`` empty, ``format`` "OPEN").
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from .base import BenchmarkSample, DialogueTurn, QAPair, infer_role, load_json, unwrap_records


class PersonaMemLoader:
    name = "personamem"

    @staticmethod
    def load(path: Union[str, Path]) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        for idx, persona in enumerate(unwrap_records(load_json(path), "data", "personas")):
            sample_id = str(persona.get("persona_id") or persona.get("id") or idx)
            turns: List[DialogueTurn] = []
            for entry in persona.get("history") or persona.get("dialogue") or []:
                speaker = str(entry.get("role") or entry.get("speaker") or "user")
                turns.append(
                    DialogueTurn(
                        index=len(turns) + 1,
                        speaker=speaker,
                        text=str(entry.get("content") or entry.get("text") or ""),
                        role=infer_role(speaker),
                        session=str(entry["session"]) if entry.get("session") is not None else None,
                        timestamp=entry.get("timestamp"),
                    )
                )
            qa_pairs: List[QAPair] = []
            for j, q in enumerate(persona.get("questions") or []):
                options = [str(o) for o in (q.get("options") or q.get("choices") or [])]
                qa_pairs.append(
                    QAPair(
                        qid=str(q.get("question_id") or f"{sample_id}::Q{j:04d}"),
                        question=str(q.get("question", "")),
                        answer=str(q.get("answer") or q.get("answer_text") or ""),
                        category=q.get("category"),
                        options=options or None,
                        meta={
                            "format": q.get("format") or ("MCQ" if options else "OPEN"),
                            "answer_text": q.get("answer_text"),
                        },
                    )
                )
            samples.append(
                BenchmarkSample(sample_id=sample_id, dialogue=turns, qa_pairs=qa_pairs, meta={"source": "personamem"})
            )
        return samples
