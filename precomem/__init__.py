"""PreCoMem: Predictive Cognitive Memory for self-evolving long-term dialogue agents.

Quick start::

    from precomem import PreCoMem, load_config

    agent = PreCoMem(load_config())            # needs OPENAI_API_KEY (or a base_url)
    agent.step("I usually start my mornings with coffee.")
    agent.step("Coffee has been making my stomach uneasy lately.")
    print(agent.render_memory())
    print(agent.query("What should I drink in the morning?"))
"""

from .agent import Backends, Evidence, PreCoMem, TurnMeta, build_backends
from .config import ConfigError, PreCoMemConfig, config_from_dict, load_config
from .gate import GateDecision, Regime
from .memory import (
    BeliefStore,
    EpisodicCache,
    EpisodicTrace,
    HypothesisClass,
    MemoryItem,
    MemoryState,
    MemoryType,
    TopicTable,
)
from .surprisal.signals import SurprisalSignal
from .trace import TurnTrace

__version__ = "1.0.0"

__all__ = [
    "Backends",
    "Evidence",
    "PreCoMem",
    "TurnMeta",
    "build_backends",
    "ConfigError",
    "PreCoMemConfig",
    "config_from_dict",
    "load_config",
    "GateDecision",
    "Regime",
    "BeliefStore",
    "EpisodicCache",
    "EpisodicTrace",
    "HypothesisClass",
    "MemoryItem",
    "MemoryState",
    "MemoryType",
    "TopicTable",
    "SurprisalSignal",
    "TurnTrace",
    "__version__",
]
