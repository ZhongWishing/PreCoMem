"""Shared offline test doubles: no network, no model downloads.

* :class:`ScriptedLLM` answers the three sub-agent prompts, the judge prompt
  and QA prompts with schema-valid JSON / text, and can be re-scripted per test.
* :class:`FixedVerifier` returns a constant contradiction probability so tests
  can steer the gate deterministically.
* :class:`FakeDiMoBenchLLM` plays both generator and verifier for the DiMoBench
  pipeline by reading the ``PLAN_JSON`` block embedded in each prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

from precomem.agent import Backends, PreCoMem
from precomem.config import PreCoMemConfig, load_config
from precomem.llm.client import LLMClient, LLMResponse, TokenUsage
from precomem.retrieval.embedder import HashingEmbedder
from precomem.surprisal.nli import ContradictionVerifier, HeuristicVerifier

# ----------------------------------------------------------------------
# LLM doubles
# ----------------------------------------------------------------------


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload)


class ScriptedLLM(LLMClient):
    """Routes prompts to canned replies by marker; records every prompt."""

    def __init__(self) -> None:
        super().__init__(model="scripted")
        self.prompts: List[str] = []
        self.systems: List[Optional[str]] = []
        self.target_id: Optional[str] = None  # id the agents should reference
        self.maintain_payload: Dict[str, Any] = {
            "reinforced_memories": [{"id": "TARGET", "evidence_strength": 0.6, "reason": "repeats the routine"}],
            "refinement_action": {"type": "none", "target_parent_id": None, "new_content": None},
        }
        self.profile_payload: Dict[str, Any] = {
            "classification": "explicit_event",
            "decision_logic": "Authorship test: the user said it.",
            "hypothesis_content": "User tried matcha at a new cafe",
            "initial_confidence": 0.65,
            "decay_protection": True,
            "evidence_keywords": ["matcha", "cafe"],
        }
        self.correct_payload: Dict[str, Any] = {
            "diagnosis": "The user no longer drinks coffee.",
            "conflict_type": "state_update",
            "contradicted_memories": [
                {"id": "TARGET", "operation": "archive_and_replace", "severity": 0.7, "reason": "explicit reversal"}
            ],
            "new_fact_content": "User prefers tea over coffee now",
        }
        self.judge_label = "CORRECT"
        self.answer_text = "Thought: based on memory.\nAnswer: tea"
        self.response_text = "Sure, here is a reply."
        self.custom: Optional[Callable[[str], Optional[str]]] = None

    def _fill(self, payload: Dict[str, Any]) -> str:
        text = _json(payload)
        return text.replace("TARGET", self.target_id or "missing")

    def complete(self, prompt, system=None, temperature=None, max_tokens=None, json_mode=False) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        usage = TokenUsage(prompt_tokens=len(prompt) // 4, completion_tokens=20)
        if self.custom is not None:
            out = self.custom(prompt)
            if out is not None:
                self._account(usage)
                return LLMResponse(text=out, usage=usage)
        if "Memory Reinforcer" in prompt:
            text = self._fill(self.maintain_payload)
        elif "Epistemic Profiler" in prompt:
            text = self._fill(self.profile_payload)
        elif "Conflict Resolver" in prompt:
            text = self._fill(self.correct_payload)
        elif "label an answer" in prompt:
            text = _json({"label": self.judge_label})
        elif "# OUTPUT FORMAT" in prompt or "# RESPONSE:" in prompt:
            text = self.answer_text
        else:
            text = self.response_text
        self._account(usage)
        return LLMResponse(text=text, usage=usage)


class FixedVerifier(ContradictionVerifier):
    name = "fixed"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def contradict_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return [self.value for _ in pairs]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def offline_config() -> PreCoMemConfig:
    return load_config(
        overrides={"embedding": {"provider": "hashing", "fallback_dim": 128}, "nli": {"model": "heuristic"}}
    )


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


def make_agent(cfg: PreCoMemConfig, llm: LLMClient, verifier: Optional[ContradictionVerifier] = None) -> PreCoMem:
    backends = Backends(llm=llm, embedder=HashingEmbedder(dim=128), verifier=verifier or HeuristicVerifier())
    return PreCoMem(cfg, backends)


@pytest.fixture
def agent(offline_config: PreCoMemConfig, scripted_llm: ScriptedLLM) -> PreCoMem:
    return make_agent(offline_config, scripted_llm)


# ----------------------------------------------------------------------
# DiMoBench double
# ----------------------------------------------------------------------

_DIALOGUE_ID_RE = re.compile(r"\bD\d+:\d+\b")


def extract_plan(prompt: str) -> Any:
    """Parse the JSON value that follows the ``PLAN_JSON`` marker."""
    idx = prompt.index("PLAN_JSON")
    start = idx + len("PLAN_JSON")
    while prompt[start] in " \n":
        start += 1
    value, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return value


class FakeDiMoBenchLLM(LLMClient):
    """Deterministic generator/verifier for the DiMoBench pipeline tests."""

    def __init__(self, model: str = "fake-generator", verdict: str = "ACCEPT", revise_first_n: int = 0) -> None:
        super().__init__(model=model)
        self.verdict = verdict
        self.revise_first_n = revise_first_n  # the first N distinct questions get REVISE on their first check
        self._seen_questions: List[str] = []
        self.calls: Dict[str, int] = {}

    def complete(self, prompt, system=None, temperature=None, max_tokens=None, json_mode=False) -> LLMResponse:
        stage = re.search(r"\[DIMOBENCH STAGE: (\w+)\]", prompt).group(1)
        self.calls[stage] = self.calls.get(stage, 0) + 1
        handler = getattr(self, f"_{stage}")
        text = _json(handler(prompt))
        self._account(TokenUsage(prompt_tokens=len(prompt) // 4, completion_tokens=50))
        return LLMResponse(text=text, usage=TokenUsage())

    # -- stages -------------------------------------------------------
    def _persona(self, prompt: str) -> Dict[str, Any]:
        plan = extract_plan(prompt)
        hints = plan.get("social_circle_hint") or ["Leo", "Leon", "Mia"]
        circle = [
            {"name": hints[0], "relation": "flatmate", "confusable_with": hints[1]},
            {"name": hints[1], "relation": "colleague", "confusable_with": hints[0]},
            {"name": hints[2] if len(hints) > 2 else "Mia", "relation": "sister", "confusable_with": None},
        ]
        return {
            "name": "Mike",
            "summary": "Mike is a sociable analyst in his twenties.",
            "speaking_style": "casual, short sentences",
            "social_circle": circle,
            "routine": ["morning coffee", "weekend KTV"],
            "drift_triggers": {str(k): f"trigger {k}" for k in range(1, len(plan.get("drift_turns", [])) + 1)},
        }

    def _skeleton(self, prompt: str) -> Dict[str, Any]:
        plan = extract_plan(prompt)
        sessions = []
        for s in plan:
            sessions.append(
                {
                    "index": s["index"],
                    "summary": f"Session {s['index']} summary ({s['phase']}).",
                    "setting": f"a few days after session {s['index'] - 1}",
                    "cues": [
                        {
                            "cue_id": c["cue_id"],
                            "description": f"behavior revealing: {c['state_to_express']}",
                            "surface_hint": "mentions weekend plans",
                        }
                        for c in s["cues"]
                    ],
                    "confounder_script": "chat about a colleague's party" if s.get("confounder") else "",
                }
            )
        return {"sessions": sessions, "chunk_summary": "Things continue."}

    def _dialogue(self, prompt: str) -> Dict[str, Any]:
        plan = extract_plan(prompt)
        n = int(re.search(r"Produce exactly (\d+)", prompt).group(1))
        turns = []
        for k in range(1, n + 1):
            speaker = "user" if k % 2 else "assistant"
            turns.append(
                {
                    "speaker": speaker,
                    "text": f"S{plan['session_index']} turn {k} ({speaker}) last weekend I did things.",
                }
            )
        user_turns = [k for k in range(1, n + 1) if k % 2 == 1]
        realizations = [
            {"cue_id": c["cue_id"], "turn": user_turns[i % len(user_turns)]} for i, c in enumerate(plan["cues"])
        ]
        return {
            "turns": turns,
            "cue_realizations": realizations,
            "confounder_turns": [user_turns[-1]] if plan.get("confounder") else [],
            "entity_decoy_turns": [user_turns[-2]] if plan.get("mention_confusable_entity") else [],
            "rolling_summary": f"Up to session {plan['session_index']}.",
        }

    def _qa(self, prompt: str) -> Dict[str, Any]:
        plan = extract_plan(prompt)
        excerpt_block = prompt.split("may be cited:")[1].split("Question requests")[0]
        ids = _DIALOGUE_ID_RE.findall(excerpt_block)
        questions = []
        for i, req in enumerate(plan):
            ev = [ids[-1 - (i % min(3, len(ids)))]] if ids else []
            questions.append(
                {
                    "qid": req["qid"],
                    "question": f"What did Mike do about {req['focus_slot']} ({req['category']}, {req['interference']})?",
                    "answer": "He kept to his usual routine.",
                    "evidence": ev,
                    "distractor_ids": [ids[0]] if req["interference"] != "none" and ids else [],
                    "rationale": "the cited turn shows it",
                }
            )
        return {"questions": questions}

    def _verify(self, prompt: str) -> Dict[str, Any]:
        question = prompt.split("The question:")[1].strip()
        decision = self.verdict
        if question not in self._seen_questions:
            self._seen_questions.append(question)
            if len(self._seen_questions) <= self.revise_first_n:
                decision = "REVISE"
        ok = decision == "ACCEPT"
        scores = {
            k: (1 if ok else 0)
            for k in (
                "answer_grounding",
                "evidence_complete",
                "temporal_consistency",
                "entity_accuracy",
                "interference_valid",
                "drift_coherent",
            )
        }
        if not ok:
            scores["answer_grounding"] = 1
        return {"scores": scores, "decision": decision, "feedback": "tighten the evidence" if not ok else ""}
