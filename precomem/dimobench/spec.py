"""Structural specification of DiMoBench and DiMoBench-XL (Appendix D.6).

Everything that the paper fixes numerically lives here, so that the LLM
stages only fill in language, never structure:

* **Base track** — one persona attitude slot; 35 sessions × 10 turns;
  sessions 1–20 (turns 1–200) baseline, the drift at turn 200 is introduced
  through 3–4 implicit events in the change-point session (turns 201–210),
  reinforced with dense cues over the next three sessions (turns 211–240),
  then a low-frequency maintenance stage (turns 241–350). Confounders every
  30 turns. 200 QA pairs: 90 pre-change, 10 change-point, 100 post-change
  (dense right after the drift, sparse in the maintenance tail).
* **XL track** — five attribute slots; 105 sessions × 10 turns; three drifts
  at turns {280, 580, 870}, each shifting one or two slots through 20–30
  behavioral cues spread over the change-point session and the three dense
  sessions that follow. 20 QA pairs per 50-turn window (21 windows, 420
  pairs), one per (reasoning category × interference tag) cell.

Phase boundaries are defined in turns (200 / 210 / 240 / 350). With 10-turn
sessions this gives the session layout 1–20 (baseline), 21 (change point),
22–24 (dense cues) and 25–35 (maintenance).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class Phase(str, Enum):
    """Session phase relative to the most recent drift."""

    PRE = "pre"
    CHANGE_POINT = "change-point"
    DENSE = "post-change-dense"
    MAINTENANCE = "post-change-maintenance"

    @property
    def qa_phase(self) -> str:
        """The three-way phase label used for QA quotas and reporting."""
        if self is Phase.PRE:
            return "pre-change"
        if self is Phase.CHANGE_POINT:
            return "change-point"
        return "post-change"


class CueKind(str, Enum):
    BASELINE = "baseline"  # expresses a slot's current (pre-drift) state
    IMPLICIT_EVENT = "implicit_event"  # one of the 3-4 events that introduce a drift
    DENSE = "dense_cue"  # reinforcing cue right after the drift
    MAINTENANCE = "maintenance_cue"  # occasional post-drift cue
    KEYWORD_TRAP = "keyword_trap"  # irrelevant chit-chat sharing surface keywords
    HYPOTHETICAL = "hypothetical"  # hypothetical narration, not a fact
    ENTITY_DECOY = "entity_decoy"  # a confusable character does/says the thing


class Category(str, Enum):
    CURRENT_STATE = "current-state inference"
    HISTORY = "history recall"
    CHANGE_POINT = "change-point detection"
    HYPOTHESIS_VS_FACT = "hypothesis-vs-fact distinction"


class Interference(str, Enum):
    ENTITY = "entity_confusable"
    KEYWORD = "keyword_trap"
    TEMPORAL = "temporal_decoy"
    HYPOTHETICAL = "hypothetical_vs_fact"
    NONE = "none"


CATEGORIES: Tuple[Category, ...] = tuple(Category)
INTERFERENCES: Tuple[Interference, ...] = tuple(Interference)
CELLS: Tuple[Tuple[Category, Interference], ...] = tuple(itertools.product(CATEGORIES, INTERFERENCES))


@dataclass(frozen=True)
class DriftPlan:
    turn: int  # the drift turn d_k; the change-point session starts at d_k + 1
    slots: Tuple[str, ...]  # attribute slots shifted by this drift
    n_implicit_events: int = 4  # cues in the change-point session (paper: 3-4)
    dense_cues_per_session: int = 5
    maintenance_cues_per_session: int = 1


@dataclass(frozen=True)
class CuePlan:
    cue_id: str
    slot: str
    kind: CueKind
    session: int
    drift_index: Optional[int] = None  # 1-based index of the drift this cue belongs to


@dataclass(frozen=True)
class SessionPlan:
    index: int  # 1-based
    first_turn: int
    last_turn: int
    phase: Phase
    drift_index: Optional[int]  # most recent drift (None before the first)
    cues: Tuple[CuePlan, ...]
    confounder: Optional[CueKind]  # KEYWORD_TRAP | HYPOTHETICAL | None
    mention_confusable_entity: bool

    @property
    def turn_ids(self) -> List[str]:
        return [f"D{self.index}:{k}" for k in range(1, self.last_turn - self.first_turn + 2)]


@dataclass(frozen=True)
class QACell:
    """One question to generate: where it is asked and what it must probe."""

    qid: str
    anchor_turn: int  # ask after this dialogue turn
    anchor_session: int
    window: int  # QA window index (session for base, 50-turn window for XL)
    qa_phase: str  # pre-change | change-point | post-change
    category: Category
    interference: Interference
    focus_slot: str
    drift_index: Optional[int]


@dataclass(frozen=True)
class DiMoBenchSpec:
    name: str
    slots: Tuple[str, ...]
    drifts: Tuple[DriftPlan, ...]
    n_sessions: int
    turns_per_session: int = 10
    dense_sessions: int = 3
    confounder_every_turns: int = 30
    baseline_cues_per_session: int = 2
    qa_window_turns: Optional[int] = None  # XL: 50-turn windows; base: None (per-session quotas)
    qa_per_window: int = 20
    base_qa_quota: Tuple[int, int, int] = (90, 10, 100)  # pre / change-point / post
    seed: int = 1234

    # ------------------------------------------------------------------
    @property
    def n_turns(self) -> int:
        return self.n_sessions * self.turns_per_session

    @property
    def drift_turns(self) -> List[int]:
        return [d.turn for d in self.drifts]

    def session_of_turn(self, turn: int) -> int:
        return (turn - 1) // self.turns_per_session + 1

    def session_bounds(self, index: int) -> Tuple[int, int]:
        first = (index - 1) * self.turns_per_session + 1
        return first, first + self.turns_per_session - 1

    def phase_of_session(self, index: int) -> Tuple[Phase, Optional[int]]:
        first, _ = self.session_bounds(index)
        current: Optional[int] = None
        for k, drift in enumerate(self.drifts, start=1):
            if first > drift.turn:
                current = k
        if current is None:
            return Phase.PRE, None
        change_session = self.session_of_turn(self.drifts[current - 1].turn) + 1
        if index == change_session:
            return Phase.CHANGE_POINT, current
        if index <= change_session + self.dense_sessions:
            return Phase.DENSE, current
        return Phase.MAINTENANCE, current

    # ------------------------------------------------------------------
    def session_plans(self) -> List[SessionPlan]:
        """Derive every session's phase, cue budget and confounder directive."""
        rng = random.Random(self.seed)
        plans: List[SessionPlan] = []
        counter = itertools.count(1)
        confounder_cycle = itertools.cycle([CueKind.KEYWORD_TRAP, CueKind.HYPOTHETICAL])
        for index in range(1, self.n_sessions + 1):
            first, last = self.session_bounds(index)
            phase, drift_index = self.phase_of_session(index)
            cues = [
                CuePlan(f"c{next(counter):04d}", slot, kind, index, d)
                for slot, kind, d in self._cue_budget(index, phase, drift_index)
            ]
            confounder = next(confounder_cycle) if last % self.confounder_every_turns == 0 else None
            plans.append(
                SessionPlan(
                    index=index,
                    first_turn=first,
                    last_turn=last,
                    phase=phase,
                    drift_index=drift_index,
                    cues=tuple(cues),
                    confounder=confounder,
                    mention_confusable_entity=confounder is not None or rng.random() < 0.15,
                )
            )
        return plans

    def _cue_budget(
        self, index: int, phase: Phase, drift_index: Optional[int]
    ) -> List[Tuple[str, CueKind, Optional[int]]]:
        """(slot, kind, drift) for every cue a session must express."""
        if phase is Phase.PRE:
            return [
                (slot, CueKind.BASELINE, None) for slot in self._rotating_slots(index, self.baseline_cues_per_session)
            ]
        drift = self.drifts[drift_index - 1]
        if phase is Phase.CHANGE_POINT:
            slots = itertools.islice(itertools.cycle(drift.slots), drift.n_implicit_events)
            return [(slot, CueKind.IMPLICIT_EVENT, drift_index) for slot in slots]
        if phase is Phase.DENSE:
            slots = itertools.islice(itertools.cycle(drift.slots), drift.dense_cues_per_session)
            return [(slot, CueKind.DENSE, drift_index) for slot in slots]
        shifted = drift.slots[(index - 1) % len(drift.slots)]
        budget = [(shifted, CueKind.MAINTENANCE, drift_index)] * drift.maintenance_cues_per_session
        # Keep the other slots observable so that stable beliefs can be probed too.
        others = [s for s in self.slots if s not in drift.slots]
        if others:
            budget.append((others[(index - 1) % len(others)], CueKind.BASELINE, None))
        return budget

    def _rotating_slots(self, session_index: int, n: int) -> List[str]:
        start = (session_index - 1) * n
        return [self.slots[(start + i) % len(self.slots)] for i in range(n)]

    # ------------------------------------------------------------------
    def qa_cells(self) -> List[QACell]:
        plans = self.session_plans()
        return self._xl_cells(plans) if self.qa_window_turns else self._base_cells(plans)

    def _base_cells(self, plans: Sequence[SessionPlan]) -> List[QACell]:
        """Per-session quotas: pre 4/5 alternating (=90), change-point 10, dense 15 each, maintenance 5 each (=100)."""
        quotas: Dict[int, int] = {}
        for p in plans:
            if p.phase is Phase.PRE:
                quotas[p.index] = 5 if p.index % 2 == 1 else 4
            elif p.phase is Phase.CHANGE_POINT:
                quotas[p.index] = 10
            elif p.phase is Phase.DENSE:
                quotas[p.index] = 15
            else:
                quotas[p.index] = 5
        cells: List[QACell] = []
        cycle = itertools.cycle(CELLS)
        for p in plans:
            for _ in range(quotas[p.index]):
                category, interference = next(cycle)
                cells.append(self._make_cell(len(cells) + 1, p, p.index, category, interference))
        return cells

    def _xl_cells(self, plans: Sequence[SessionPlan]) -> List[QACell]:
        """One question per (category, interference) cell in every 50-turn window."""
        assert self.qa_window_turns is not None
        by_index = {p.index: p for p in plans}
        cells: List[QACell] = []
        n_windows = self.n_turns // self.qa_window_turns
        for w in range(1, n_windows + 1):
            anchor = w * self.qa_window_turns
            plan = by_index[self.session_of_turn(anchor)]
            grid = list(CELLS)
            random.Random(self.seed + w).shuffle(grid)
            for category, interference in grid[: self.qa_per_window]:
                cells.append(self._make_cell(len(cells) + 1, plan, w, category, interference, anchor_turn=anchor))
        return cells

    def _make_cell(
        self,
        number: int,
        plan: SessionPlan,
        window: int,
        category: Category,
        interference: Interference,
        anchor_turn: Optional[int] = None,
    ) -> QACell:
        anchor = anchor_turn or plan.last_turn
        anchor_session = self.session_of_turn(anchor)
        # Interference that needs material not yet present is downgraded to "none".
        first_keyword_trap = self.confounder_every_turns  # first confounder session ends here
        first_hypothetical = 2 * self.confounder_every_turns  # the cycle alternates keyword / hypothetical
        if interference is Interference.KEYWORD and anchor < first_keyword_trap:
            interference = Interference.NONE
        elif interference is Interference.HYPOTHETICAL and anchor < first_hypothetical:
            interference = Interference.NONE
        elif interference is Interference.TEMPORAL and anchor_session < 2:
            interference = Interference.NONE
        qa_phase = self._qa_phase_at(anchor)
        drift_index = plan.drift_index if qa_phase != "pre-change" else None
        focus = self._focus_slot(number, drift_index)
        return QACell(
            qid=f"Q{number:04d}",
            anchor_turn=anchor,
            anchor_session=anchor_session,
            window=window,
            qa_phase=qa_phase,
            category=category,
            interference=interference,
            focus_slot=focus,
            drift_index=drift_index,
        )

    def _qa_phase_at(self, anchor_turn: int) -> str:
        session = self.session_of_turn(anchor_turn)
        phase, _ = self.phase_of_session(session)
        if self.qa_window_turns:
            # A window that contains a change-point session is a change-point window.
            lo = anchor_turn - self.qa_window_turns + 1
            for s in range(self.session_of_turn(lo), session + 1):
                if self.phase_of_session(s)[0] is Phase.CHANGE_POINT:
                    return "change-point"
        return phase.qa_phase

    def _focus_slot(self, number: int, drift_index: Optional[int]) -> str:
        if drift_index is None:
            return self.slots[(number - 1) % len(self.slots)] if len(self.slots) > 1 else self.slots[0]
        shifted = self.drifts[drift_index - 1].slots
        if len(self.slots) == 1:
            return self.slots[0]
        # Post-drift: alternate between the shifted slot(s) and a stable slot (old-memory stability probe).
        stable = [s for s in self.slots if s not in shifted] or list(shifted)
        return shifted[number % len(shifted)] if number % 2 == 0 else stable[number % len(stable)]

    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, object]:
        plans = self.session_plans()
        cells = self.qa_cells()
        by_phase: Dict[str, int] = {}
        for c in cells:
            by_phase[c.qa_phase] = by_phase.get(c.qa_phase, 0) + 1
        return {
            "name": self.name,
            "n_sessions": self.n_sessions,
            "n_turns": self.n_turns,
            "drift_turns": self.drift_turns,
            "slots": list(self.slots),
            "n_cues": sum(len(p.cues) for p in plans),
            "n_confounder_sessions": sum(1 for p in plans if p.confounder),
            "n_qa": len(cells),
            "qa_by_phase": by_phase,
        }


# ----------------------------------------------------------------------
# The two released tracks
# ----------------------------------------------------------------------

XL_SLOTS: Tuple[str, ...] = ("drink_preference", "exercise_habit", "home_city", "employment_status", "social_style")


def base_spec(core_slot: str, seed: int = 1234) -> DiMoBenchSpec:
    return DiMoBenchSpec(
        name="dimobench",
        slots=(core_slot,),
        drifts=(DriftPlan(turn=200, slots=(core_slot,), n_implicit_events=4, dense_cues_per_session=5),),
        n_sessions=35,
        seed=seed,
    )


def xl_spec(
    drift_slots: Sequence[Sequence[str]] = (("social_style",), ("drink_preference", "exercise_habit"), ("home_city",)),
    seed: int = 1234,
) -> DiMoBenchSpec:
    """Three drifts at {280, 580, 870}; each shifts one or two of the five slots.

    With 4 implicit events + 3 × 6 dense cues each drift is expressed through
    22 cues, inside the 20–30 band of the paper.
    """
    if len(drift_slots) != 3:
        raise ValueError("DiMoBench-XL has exactly three drifts")
    drifts = tuple(
        DriftPlan(turn=t, slots=tuple(s), n_implicit_events=4, dense_cues_per_session=6)
        for t, s in zip((280, 580, 870), drift_slots)
    )
    return DiMoBenchSpec(
        name="dimobench_xl",
        slots=XL_SLOTS,
        drifts=drifts,
        n_sessions=105,
        qa_window_turns=50,
        qa_per_window=20,
        seed=seed,
    )
