"""The PreCoMem agent: one pass through Algorithm 1 per dialogue turn.

    Phase A  Memory Leveraging (deterministic)
        (C_t, s_t) <- Retrieve(M_t, u_t)                          Eq. (1)
        y_t        <- Respond(u_t, Q_{t-1}, C_t)                  Eq. (2)
    Phase B  Reliability + surprisal (deterministic)
        rho_t, d_sem, d_conf, S_raw, x_t = rho_t * S_raw          Eqs. (3)-(5)
    Phase C  Regime-gated update (LLM sub-agent + closed-form rule)
        MAINTAIN / PROFILE / CORRECT                             Eqs. (6)-(10)
    Phase D  Hypothesis lifecycle + episodic write-back          Eq. (9)

:meth:`PreCoMem.step` generates the response with the backbone;
:meth:`PreCoMem.observe` replays a recorded exchange (the benchmark setting)
so that the write path runs without the response-generation call. Both go
through the same :meth:`_run_turn`.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

from .config import PreCoMemConfig, load_config
from .gate import GateDecision, Regime, ThresholdPolicy, build_threshold_policy, gate
from .llm.client import LLMClient, MeteredLLM, TokenUsage, build_backbone_client
from .llm.prompts import RESPONSE_PROMPT
from .memory.items import EpisodicTrace, MemoryItem
from .memory.store import EpisodicCache, MemoryState, TopicTable
from .operators.base import NoOpOutcome, OperatorOutcome
from .operators.correct import CorrectOperator
from .operators.lifecycle import HypothesisLifecycle
from .operators.maintain import MaintainOperator
from .operators.profile import ProfileOperator
from .retrieval.embedder import Embedder, build_embedder
from .retrieval.retriever import RetrievalResult, Retriever
from .surprisal.nli import ContradictionVerifier, build_verifier
from .surprisal.signals import SurprisalSignal, compute_surprisal
from .trace import TurnTrace, write_traces
from .utils.logging import get_logger

_LOGGER = get_logger("agent")


@dataclass
class Backends:
    """The three heavy, shareable components. Build once, pass to many agents."""

    llm: LLMClient
    embedder: Embedder
    verifier: ContradictionVerifier


def build_backends(cfg: PreCoMemConfig, llm: Optional[LLMClient] = None) -> Backends:
    return Backends(
        llm=llm or build_backbone_client(cfg.backbone),
        embedder=build_embedder(cfg.embedding),
        verifier=build_verifier(cfg.nli),
    )


@dataclass(frozen=True)
class TurnMeta:
    """Optional benchmark metadata attached to the episodic trace of a turn."""

    speaker: Optional[str] = None
    session: Optional[str] = None
    dialogue_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class Evidence:
    """What a read-only query recalls, rendered for a prompt.

    ``beliefs`` is ``C_t`` (Top-K), ``excerpts`` the episodic traces linked from
    it (at most ``retrieval.max_provenance_traces``), and ``recent`` the rolling
    working-memory window ``Q_{t-1}``. Together they are exactly the memory
    part of the response-model input whose token count the paper reports.
    """

    retrieval: RetrievalResult
    beliefs: List[MemoryItem] = field(default_factory=list)
    excerpts: List[EpisodicTrace] = field(default_factory=list)
    recent: List[Tuple[str, str]] = field(default_factory=list)

    def render(self) -> str:
        """Plain-text context. Confidence weights are deliberately not shown."""
        parts: List[str] = []
        if self.beliefs:
            parts.append("Consolidated beliefs about the user (most relevant first):")
            parts.extend(f"- {m.content}" for m in self.beliefs)
        if self.excerpts:
            parts.append("\nSupporting dialogue excerpts (chronological):")
            parts.extend(t.render() for t in self.excerpts)
        if self.recent:
            parts.append("\nMost recent turns:")
            parts.extend(f"User: {u}\nAssistant: {a}" if a else f"User: {u}" for u, a in self.recent)
        return "\n".join(parts) if parts else "(no memory available)"


class PreCoMem:
    """Predictive Cognitive Memory for one user / one dialogue."""

    def __init__(
        self,
        config: Optional[PreCoMemConfig] = None,
        backends: Optional[Backends] = None,
        *,
        llm: Optional[LLMClient] = None,
        embedder: Optional[Embedder] = None,
        verifier: Optional[ContradictionVerifier] = None,
        seed: int = 0,
    ) -> None:
        self.config = config or load_config()
        cfg = self.config
        if backends is None:
            backends = Backends(
                llm=llm or build_backbone_client(cfg.backbone),
                embedder=embedder or build_embedder(cfg.embedding),
                verifier=verifier or build_verifier(cfg.nli),
            )
        self.backends = backends
        # Each agent meters its own token usage even when the provider client is shared.
        self.llm: LLMClient = MeteredLLM(backends.llm)
        self.embedder, self.verifier = backends.embedder, backends.verifier

        ab = cfg.ablation
        self.ablation = ab
        self.retriever = Retriever(self.embedder, cfg.retrieval, use_weights=ab.read_time_weighting and not ab.cw)
        self.maintain = MaintainOperator(self.llm, cfg.operators.maintain)
        self.profile = ProfileOperator(
            self.llm,
            cfg.operators.profile,
            direct_commit=ab.without_profile,
            commit_weight=cfg.operators.correct.new_fact_weight,
        )
        self.correct = CorrectOperator(self.llm, cfg.operators.correct)
        self.lifecycle = HypothesisLifecycle(cfg.lifecycle)
        self.thresholds: ThresholdPolicy = build_threshold_policy(cfg.gate)
        self._routing_rng = random.Random(seed)  # used only by the random-routing control
        if ab.any_active:
            _LOGGER.warning("Ablation switches active: %s", ab.active_switches())

        self.state = self._fresh_state()
        self.short_term: Deque[Tuple[str, str]] = deque(maxlen=cfg.retrieval.short_term_window)
        self.history: List[TurnTrace] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget this user: empty memory, empty working window, default thresholds."""
        self.state = self._fresh_state()
        self.short_term.clear()
        self.history.clear()
        self.thresholds.reset()

    def step(self, user_input: str, update_memory: bool = True) -> TurnTrace:
        """Full turn: retrieve, *generate* the response, then evolve memory."""
        return self._run_turn(user_input, response=None, meta=None, update_memory=update_memory)

    def chat(self, user_input: str) -> str:
        return self.step(user_input).response

    def observe(
        self,
        user: str,
        assistant: str = "",
        meta: Optional[TurnMeta] = None,
        update_memory: bool = True,
    ) -> TurnTrace:
        """Replay a recorded exchange: the assistant reply is given, not generated."""
        return self._run_turn(user, response=assistant or "", meta=meta, update_memory=update_memory)

    def recall(self, question: str, top_k: Optional[int] = None, include_recent: bool = True) -> Evidence:
        """Read-only retrieval (no memory write, no turn advance)."""
        retrieval = self._retrieve(question, top_k=top_k)
        return Evidence(
            retrieval=retrieval,
            beliefs=list(retrieval.items),
            excerpts=list(retrieval.traces),
            recent=list(self.short_term) if include_recent else [],
        )

    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """Read-only retrieval-augmented answer with the generic response prompt."""
        return self._respond(question, self.recall(question, top_k=top_k).retrieval)

    def ingest_episode(self, user: str, assistant: str = "", meta: Optional[TurnMeta] = None) -> EpisodicTrace:
        """Append a turn to the episodic cache only (no belief update, no LLM call)."""
        self.state.advance()
        return self._write_episode(user, assistant, meta)

    # -- inspection / persistence --------------------------------------
    def snapshot(self, include_embeddings: bool = False) -> Dict:
        return self.state.to_dict(include_embeddings=include_embeddings)

    def render_memory(self, max_items: Optional[int] = None) -> str:
        return self.state.render(max_items=max_items)

    def save_memory(self, path: Union[str, Path], include_embeddings: bool = False) -> Path:
        return self.state.save(path, include_embeddings=include_embeddings)

    def load_memory(self, path: Union[str, Path]) -> None:
        self.state = MemoryState.load(path)
        self.state.topic_table.new_threshold = self.config.retrieval.topic_new_threshold

    def dump_history(self, path: Union[str, Path]) -> Path:
        return write_traces(self.history, path)

    # ------------------------------------------------------------------
    # Algorithm 1
    # ------------------------------------------------------------------
    def _run_turn(
        self,
        user_input: str,
        response: Optional[str],
        meta: Optional[TurnMeta],
        update_memory: bool,
    ) -> TurnTrace:
        turn = self.state.advance()
        timings: Dict[str, float] = {}
        usage_before = TokenUsage(self.llm.usage.prompt_tokens, self.llm.usage.completion_tokens)

        # Phase A: Memory Leveraging
        with _timer(timings, "retrieve"):
            retrieval = self._retrieve(user_input)
        if response is None:
            with _timer(timings, "respond"):
                response = self._respond(user_input, retrieval)

        # Phase B: reliability + surprisal
        with _timer(timings, "surprisal"):
            signal = compute_surprisal(
                user_input,
                retrieval,
                self.verifier,
                lam=self.config.surprisal.lambda_,
                eps=self.config.surprisal.reliability_eps,
                force_rho_one=self.ablation.rr,
            )

        # Phase C: regime-gated update
        decision = self._decide(signal, retrieval, user_input, update_memory)
        outcome: OperatorOutcome = NoOpOutcome(reason="read-only turn")
        lifecycle_stats: Dict[str, int] = {}
        trace_obj: Optional[EpisodicTrace] = None
        if update_memory:
            with _timer(timings, "operator"):
                outcome = self._evolve(decision, user_input, retrieval, turn)
                self._apply_cw()

            # Phase D: lifecycle + write-back
            with _timer(timings, "lifecycle"):
                stats = self.lifecycle.step(self.state.belief_store, turn)
                self._apply_cw()
                lifecycle_stats = stats.to_dict()
                trace_obj = self._write_episode(user_input, response, meta)
                self._link_provenance(outcome, trace_obj)
                self._index_new_items(outcome)
                for item in list(stats.removed) + outcome.retired():
                    self.thresholds.observe_lifespan(turn - item.created_turn)
            self.thresholds.observe_turn(decision.regime, signal.s_eff)

        self.short_term.append((user_input, response))

        trace = TurnTrace(
            turn=turn,
            user_input=user_input,
            response=response,
            gate=decision,
            surprisal=signal,
            retrieved=retrieval.summary(),
            operator=outcome.to_dict(),
            lifecycle=lifecycle_stats,
            memory=self.state.summary(),
            latency_ms=timings,
            tokens=_usage_delta(usage_before, self.llm.usage),
            trace_id=trace_obj.trace_id if trace_obj is not None else "",
        )
        self.history.append(trace)
        return trace

    def _retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return self.retriever.retrieve(
            query,
            store=self.state.belief_store,
            episodic_cache=self.state.episodic_cache,
            topic_table=self.state.topic_table,
            top_k=top_k,
        )

    def _decide(
        self, signal: SurprisalSignal, retrieval: RetrievalResult, user_input: str, update_memory: bool
    ) -> GateDecision:
        theta_low, theta_high = self.thresholds.current()
        if not update_memory or not user_input.strip():
            return GateDecision(Regime.NONE, signal.s_eff, theta_low, theta_high)
        if retrieval.is_empty:
            # Cold start (Appendix C.2): with no evidence the surprisal terms are
            # undefined, so first observations enter the store as hypotheses.
            return GateDecision(Regime.PROFILE, signal.s_eff, theta_low, theta_high, cold_start=True)
        ab = self.ablation
        x_t = 0.0 if ab.es else signal.s_eff  # ES: the gate score is forced to zero
        if ab.random_routing is not None:
            regime = self._routing_rng.choices(
                [Regime.MAINTAIN, Regime.PROFILE, Regime.CORRECT], weights=list(ab.random_routing)
            )[0]
        else:
            regime = gate(x_t, theta_low, theta_high, single_threshold=ab.single_threshold)
        return GateDecision(regime, x_t, theta_low, theta_high)

    def _evolve(
        self, decision: GateDecision, user_input: str, retrieval: RetrievalResult, turn: int
    ) -> OperatorOutcome:
        ab = self.ablation
        removed = {
            Regime.MAINTAIN: ab.without_maintain,
            Regime.PROFILE: False,  # "w/o PROFILE" is handled by ProfileOperator.direct_commit
            Regime.CORRECT: ab.without_correct,
        }
        if removed.get(decision.regime, False):
            return NoOpOutcome(reason=f"{decision.regime.value} disabled by ablation")
        operator = {
            Regime.MAINTAIN: self.maintain,
            Regime.PROFILE: self.profile,
            Regime.CORRECT: self.correct,
        }.get(decision.regime)
        if operator is None:
            return NoOpOutcome(reason="no update")
        return operator(user_input, retrieval, decision.x_t, self.state.belief_store, turn)

    def _apply_cw(self) -> None:
        """CW ablation: overwrite every belief weight with 1 (retrieval unweighted, lifecycle trivial)."""
        if not self.ablation.cw:
            return
        for item in self.state.belief_store.items():
            item.weight = 1.0

    # ------------------------------------------------------------------
    # Write-back helpers
    # ------------------------------------------------------------------
    def _write_episode(self, user_input: str, response: str, meta: Optional[TurnMeta]) -> EpisodicTrace:
        meta = meta or TurnMeta()
        trace = EpisodicTrace(
            trace_id=f"E{self.state.turn:06d}",
            turn=self.state.turn,
            user=user_input,
            assistant=response or "",
            speaker=meta.speaker,
            session=meta.session,
            dialogue_id=meta.dialogue_id,
            timestamp=meta.timestamp,
        )
        return self.state.episodic_cache.append(trace)

    def _link_provenance(self, outcome: OperatorOutcome, trace: EpisodicTrace) -> None:
        store = self.state.belief_store
        for item_id in outcome.written_ids():
            item = store.get(item_id)
            if item is not None:
                item.add_source(trace.trace_id)

    def _index_new_items(self, outcome: OperatorOutcome) -> None:
        """Embed items written this turn and assign their topic label ``z(m)``."""
        store = self.state.belief_store
        fresh = [store.get(i) for i in outcome.written_ids()]
        fresh = [m for m in fresh if m is not None and m.topic is None]
        if not fresh:
            return
        embeddings = self.retriever.ensure_embeddings(fresh)
        table: TopicTable = self.state.topic_table
        for item, vec in zip(fresh, embeddings):
            hint = item.keywords[0] if item.keywords else " ".join(item.content.split()[:3])
            item.topic = table.assign(vec, hint)

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------
    def _respond(self, user_input: str, retrieval: RetrievalResult) -> str:
        prompt = RESPONSE_PROMPT.format(
            short_term_window=self._format_short_term(),
            retrieved_beliefs="\n".join(f"- {m.content}" for m in retrieval.items) or "(no relevant beliefs retrieved)",
            episodic_excerpts="\n".join(t.render() for t in retrieval.traces) or "(no episodic excerpts available)",
            user_input=user_input,
        )
        return self.llm.complete(prompt).text.strip()

    def _format_short_term(self) -> str:
        if not self.short_term:
            return "(no prior turns)"
        return "\n\n".join(f"User: {u}\nAssistant: {a}" if a else f"User: {u}" for u, a in self.short_term)

    def _fresh_state(self) -> MemoryState:
        return MemoryState(
            episodic_cache=EpisodicCache(capacity=self.config.retrieval.episodic_capacity),
            topic_table=TopicTable(new_threshold=self.config.retrieval.topic_new_threshold),
        )


class _timer:
    def __init__(self, sink: Dict[str, float], key: str) -> None:
        self.sink, self.key = sink, key

    def __enter__(self) -> _timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.sink[self.key] = (time.perf_counter() - self.t0) * 1000.0


def _usage_delta(before: TokenUsage, after: TokenUsage) -> Dict[str, int]:
    return {
        "prompt_tokens": after.prompt_tokens - before.prompt_tokens,
        "completion_tokens": after.completion_tokens - before.completion_tokens,
    }
