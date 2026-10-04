"""Typed configuration for PreCoMem.

The YAML file ``precomem/configs/default.yaml`` is the single source of truth for every
hyperparameter. It is parsed into the frozen dataclasses below so that

* each field has a type and a documented default,
* unknown keys raise :class:`ConfigError` instead of being ignored, and
* the structural invariants assumed by the paper are checked once, at load
  time (for example ``0 <= w_min < w_max < tau_promote <= 1`` for both
  hypothesis bands, ``theta_low < theta_high``, convex retrieval weights).

Use :func:`load_config` to read the defaults, optionally merged with a user
YAML and a dictionary of programmatic overrides::

    cfg = load_config("my_run.yaml", overrides={"backbone": {"model": "gpt-4.1"}})
    cfg.gate.theta_high  # -> 0.6
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types, or violated invariants."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: float = 60.0
    max_retries: int = 6
    json_mode: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "huggingface"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "auto"
    batch_size: int = 32
    max_seq_len: int = 512
    fallback_dim: int = 256


@dataclass(frozen=True)
class NLIConfig:
    model: str = "cross-encoder/nli-deberta-v3-base"
    device: str = "auto"
    temperature: float = 1.0
    batch_size: int = 16
    max_length: int = 256


@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 0.4
    bm25: float = 0.4
    topic: float = 0.2

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.semantic, self.bm25, self.topic)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    alpha: RetrievalWeights = field(default_factory=RetrievalWeights)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    topic_new_threshold: float = 0.5
    short_term_window: int = 6
    episodic_capacity: int = 20000
    max_provenance_traces: int = 20


@dataclass(frozen=True)
class SurprisalConfig:
    # ``lambda`` is a Python keyword, so the YAML key ``lambda`` maps to ``lambda_``.
    lambda_: float = field(default=0.5, metadata={"yaml_key": "lambda"})
    reliability_eps: float = 1e-9


@dataclass(frozen=True)
class GateConfig:
    theta_low: float = 0.2
    theta_high: float = 0.6
    adaptive: bool = False
    adaptive_window: int = 50
    adaptive_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    adaptive_low_range: Tuple[float, float] = (0.10, 0.35)
    adaptive_high_range: Tuple[float, float] = (0.40, 0.80)
    adaptive_ema: float = 0.1
    adaptive_kappa: float = 1.0
    adaptive_ref_lifespan: int = 50


@dataclass(frozen=True)
class MaintainConfig:
    eta_min: float = 0.05
    eta_max: float = 1.0


@dataclass(frozen=True)
class ProfileConfig:
    explicit_band: Tuple[float, float] = (0.60, 0.70)
    latent_band: Tuple[float, float] = (0.50, 0.59)


@dataclass(frozen=True)
class CorrectConfig:
    beta: float = 1.0
    new_fact_weight: float = 0.85


@dataclass(frozen=True)
class OperatorConfig:
    maintain: MaintainConfig = field(default_factory=MaintainConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    correct: CorrectConfig = field(default_factory=CorrectConfig)


@dataclass(frozen=True)
class LifecycleConfig:
    promote_threshold: float = 0.80
    hypothesis_prune: float = 0.10
    decay_gamma: float = 0.05
    fact_optional_prune: Optional[float] = None


@dataclass(frozen=True)
class AblationConfig:
    """Switches for the ablations and controls reported in the paper (all off by default).

    es:                  gate score forced to 0, every evidence-bearing turn routes to MAINTAIN (Fig. 4a "ES")
    rr:                  retrieval reliability rho_t = 1 (Fig. 4a "RR"; "w/o C1" in the lifecycle table of Appendix D.3)
    cw:                  every belief weight overwritten to 1 after each write, so retrieval is unweighted,
                         hypotheses are promoted at their first lifecycle step and pruning never fires (Fig. 4a "CW")
    read_time_weighting: factorial control (i): set false to drop w_t(m) from Eq. (1) only
    without_maintain / without_profile / without_correct:
                         operator ablations of the DiMoBench study; "without_profile" commits medium-surprisal
                         observations directly as FACTs (no waiting room, "w/o C2/C3")
    single_threshold:    route the medium band to CORRECT (no PROFILE band)
    random_routing:      frequency-matched random routing, e.g. [0.562, 0.274, 0.164] for MAINTAIN/PROFILE/CORRECT
    """

    es: bool = False
    rr: bool = False
    cw: bool = False
    read_time_weighting: bool = True
    without_maintain: bool = False
    without_profile: bool = False
    without_correct: bool = False
    single_threshold: bool = False
    random_routing: Optional[Tuple[float, float, float]] = None

    def active_switches(self) -> Dict[str, Any]:
        """Switches that differ from the main-results setting."""
        defaults = AblationConfig()
        return {
            f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(defaults, f.name)
        }

    @property
    def any_active(self) -> bool:
        return bool(self.active_switches())


@dataclass(frozen=True)
class EvaluationConfig:
    primary_judge: str = "gpt-4.1-mini"
    aux_judges: Tuple[str, ...] = ("gemini-2.5-flash", "gemini-2.5-flash")
    seeds: Tuple[int, ...] = (13, 42, 1234)
    num_workers: int = 4


@dataclass(frozen=True)
class PreCoMemConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    surprisal: SurprisalConfig = field(default_factory=SurprisalConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    operators: OperatorConfig = field(default_factory=OperatorConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> PreCoMemConfig:
        g, r, o, lc = self.gate, self.retrieval, self.operators, self.lifecycle
        if not 0.0 <= g.theta_low < g.theta_high:
            raise ConfigError(
                f"gate thresholds must satisfy 0 <= theta_low < theta_high, got ({g.theta_low}, {g.theta_high})"
            )
        if any(a < 0 for a in r.alpha.as_tuple()) or sum(r.alpha.as_tuple()) <= 0:
            raise ConfigError("retrieval.alpha weights must be non-negative with positive sum")
        if r.top_k < 1:
            raise ConfigError("retrieval.top_k must be >= 1")
        if not 0.0 <= self.surprisal.lambda_ <= 1.0:
            raise ConfigError("surprisal.lambda must lie in [0, 1]")
        for name, band in (("explicit", o.profile.explicit_band), ("latent", o.profile.latent_band)):
            lo, hi = band
            if not 0.0 <= lo < hi < lc.promote_threshold <= 1.0:
                raise ConfigError(
                    f"profile.{name}_band must satisfy 0 <= w_min < w_max < tau_promote <= 1, "
                    f"got {band} with tau_promote={lc.promote_threshold}"
                )
        if not 0.0 < o.maintain.eta_min <= o.maintain.eta_max <= 1.0:
            raise ConfigError("maintain.eta_min / eta_max must satisfy 0 < eta_min <= eta_max <= 1")
        if o.correct.beta <= 0:
            raise ConfigError("correct.beta must be positive")
        if not 0.0 <= o.correct.new_fact_weight <= 1.0:
            raise ConfigError("correct.new_fact_weight must lie in [0, 1]")
        if lc.decay_gamma < 0:
            raise ConfigError("lifecycle.decay_gamma must be non-negative")
        if not 0.0 <= lc.hypothesis_prune < lc.promote_threshold:
            raise ConfigError("lifecycle.hypothesis_prune must be below promote_threshold")
        ab = self.ablation
        if ab.random_routing is not None:
            if len(ab.random_routing) != 3 or any(p < 0 for p in ab.random_routing) or sum(ab.random_routing) <= 0:
                raise ConfigError("ablation.random_routing must be three non-negative routing frequencies")
            if ab.es or ab.single_threshold:
                raise ConfigError("ablation.random_routing cannot be combined with es or single_threshold")
        if ab.es and ab.single_threshold:
            raise ConfigError("ablation.es and ablation.single_threshold are mutually exclusive")
        if g.adaptive:
            if len(g.adaptive_weights) != 3 or any(w < 0 for w in g.adaptive_weights):
                raise ConfigError("gate.adaptive_weights must be three non-negative numbers")
            for rng_name, rng in (("low", g.adaptive_low_range), ("high", g.adaptive_high_range)):
                if not rng[0] < rng[1]:
                    raise ConfigError(f"gate.adaptive_{rng_name}_range must be increasing")
            if not 0.0 < g.adaptive_ema <= 1.0:
                raise ConfigError("gate.adaptive_ema must lie in (0, 1]")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)

    def with_overrides(self, overrides: Mapping[str, Any]) -> PreCoMemConfig:
        merged = deep_merge(self.to_dict(), overrides)
        return config_from_dict(merged)


# ---------------------------------------------------------------------------
# (De)serialization helpers
# ---------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``base`` recursively updated with ``extra`` (neither is mutated)."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _yaml_key(f: dataclasses.Field) -> str:
    return f.metadata.get("yaml_key", f.name)


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Coerce YAML scalars/containers into the annotated field type."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin is Union:  # Optional[X]
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, non_none[0], path)
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping")
        return _build(annotation, value, path)
    if origin in (dict, Dict):
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping")
        return copy.deepcopy(dict(value))
    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if args and len(args) != len(value):
            raise ConfigError(f"{path}: expected {len(args)} values, got {len(value)}")
        return tuple(_coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args or [Any] * len(value))))
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected a boolean, got {value!r}")
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _build(cls: Type[T], data: Mapping[str, Any], path: str = "") -> T:
    # Field annotations are strings under ``from __future__ import annotations``;
    # ``get_type_hints`` resolves them to real types for ``_coerce``.
    hints = typing.get_type_hints(cls)
    known = {_yaml_key(f): f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: unknown key(s) {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for key, f in known.items():
        if key in data:
            kwargs[f.name] = _coerce(data[key], hints[f.name], f"{path}.{key}" if path else key)
    return cls(**kwargs)


def _to_plain_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {_yaml_key(f): _to_plain_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_to_plain_dict(v) for v in obj]
    if isinstance(obj, dict):
        return copy.deepcopy(obj)
    return obj


def config_from_dict(data: Mapping[str, Any]) -> PreCoMemConfig:
    """Build and validate a config from a plain (YAML-shaped) mapping."""
    return _build(PreCoMemConfig, data).validate()


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    defaults: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> PreCoMemConfig:
    """Load ``defaults``, merge an optional user YAML and overrides, validate."""
    with Path(defaults).open("r", encoding="utf-8") as fh:
        merged: Dict[str, Any] = yaml.safe_load(fh) or {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            merged = deep_merge(merged, yaml.safe_load(fh) or {})
    if overrides:
        merged = deep_merge(merged, overrides)
    return config_from_dict(merged)


def parse_override(expr: str) -> Dict[str, Any]:
    """Parse ``section.key=value`` (dotted path) into a nested override dict.

    Values are parsed as YAML so ``true``, ``0.6``, ``[0.5, 0.3]`` and ``null``
    all work from the command line.
    """
    if "=" not in expr:
        raise ConfigError(f"override must look like section.key=value, got {expr!r}")
    dotted, raw = expr.split("=", 1)
    value = yaml.safe_load(raw)
    out: Dict[str, Any] = {}
    cursor = out
    keys = [k for k in dotted.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"empty key in override {expr!r}")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value
    return out


def merge_overrides(exprs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for expr in exprs:
        out = deep_merge(out, parse_override(expr))
    return out
