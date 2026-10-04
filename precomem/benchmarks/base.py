"""Common benchmark types.

Every loader returns :class:`BenchmarkSample` objects with a chronological
list of :class:`DialogueTurn` and a list of :class:`QAPair`, so the runner
is benchmark-agnostic. :func:`pair_exchanges` turns the flat turn list into
the (user, assistant) exchanges PreCoMem consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

USER_ROLES = {"user", "human", "speaker_a", "u"}
ASSISTANT_ROLES = {"assistant", "ai", "bot", "system", "speaker_b", "a"}


def infer_role(speaker: str, default: str = "user") -> str:
    """Map a raw speaker label to ``user`` or ``assistant``.

    Named human speakers (LoCoMo's "Caroline", "Melanie") are treated as
    users: each of their messages is an observation the memory system should
    learn from, and the name is kept for attribution.
    """
    low = (speaker or "").strip().lower()
    if low in ASSISTANT_ROLES:
        return "assistant"
    if low in USER_ROLES:
        return "user"
    return default


@dataclass
class DialogueTurn:
    index: int  # 1-based position in the full dialogue
    speaker: str  # raw label as found in the data
    text: str
    role: str = "user"  # "user" | "assistant"
    session: Optional[str] = None
    timestamp: Optional[str] = None
    dialogue_id: Optional[str] = None  # e.g. LoCoMo "D1:3"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def named_speaker(self) -> Optional[str]:
        """The speaker name when it is a real name rather than a role label."""
        low = self.speaker.strip().lower()
        if not low or low in USER_ROLES or low in ASSISTANT_ROLES:
            return None
        return self.speaker.strip()


@dataclass
class QAPair:
    qid: str
    question: str
    answer: str
    category: Optional[str] = None
    options: Optional[List[str]] = None
    evidence: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    sample_id: str
    dialogue: List[DialogueTurn]
    qa_pairs: List[QAPair]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.dialogue)


@dataclass
class Exchange:
    """One write-path observation: a user message and the reply it received."""

    user: str
    assistant: str = ""
    speaker: Optional[str] = None
    session: Optional[str] = None
    timestamp: Optional[str] = None
    dialogue_id: Optional[str] = None
    turn_indices: List[int] = field(default_factory=list)

    def memory_input(self) -> str:
        """Text handed to the write path; named speakers are prefixed for attribution."""
        if self.speaker and self.user:
            return f"{self.speaker}: {self.user}"
        return self.user


def pair_exchanges(turns: List[DialogueTurn]) -> List[Exchange]:
    """Group turns into exchanges.

    A user turn opens an exchange; the next assistant turn closes it. Two
    consecutive user turns (or a dialogue between two humans) yield separate
    exchanges with an empty reply, and an assistant turn with no preceding
    user turn becomes an exchange with an empty user message so that the
    assistant's statement still reaches the episodic cache.
    """
    exchanges: List[Exchange] = []
    open_exchange: Optional[Exchange] = None
    for turn in turns:
        if turn.is_user:
            if open_exchange is not None:
                exchanges.append(open_exchange)
            open_exchange = Exchange(
                user=turn.text,
                speaker=turn.named_speaker,
                session=turn.session,
                timestamp=turn.timestamp,
                dialogue_id=turn.dialogue_id,
                turn_indices=[turn.index],
            )
        else:
            if open_exchange is None:
                open_exchange = Exchange(
                    user="",
                    session=turn.session,
                    timestamp=turn.timestamp,
                    dialogue_id=turn.dialogue_id,
                )
            open_exchange.assistant = turn.text
            open_exchange.turn_indices.append(turn.index)
            exchanges.append(open_exchange)
            open_exchange = None
    if open_exchange is not None:
        exchanges.append(open_exchange)
    return exchanges


def load_json(path: Union[str, Path]) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def unwrap_records(data: Any, *keys: str) -> List[Any]:
    """Accept ``[...]``, ``{"data": [...]}`` or a single record."""
    if isinstance(data, dict):
        for key in keys:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported benchmark file layout: {type(data).__name__}")
