<div align="center">

# PreCoMem: Predictive Cognitive Memory for Self-Evolving Long-Term Dialogue Agents

**NeurIPS 2026**

English | [简体中文](README_zh.md)

![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![tests](https://github.com/ZhongWishing/PreCoMem/actions/workflows/tests.yml/badge.svg)](https://github.com/ZhongWishing/PreCoMem/actions/workflows/tests.yml)

Jian Zhong<sup>1*</sup>, Zeyu Liu<sup>1*</sup>, Pingchuan Cao<sup>1</sup>, Rongduo Han<sup>1</sup>, Shunye Tang<sup>1</sup>, Guohuan Xie<sup>3</sup>, Mingyuan Qin<sup>4</sup>, Ran Ji<sup>5</sup>, Xiao Liang<sup>2</sup>, Wei Wang<sup>1†</sup>, Haining Zhang<sup>1†</sup>

<sup>1</sup>Nankai University &nbsp; <sup>2</sup>vivo AI Lab &nbsp; <sup>3</sup>Tsinghua University &nbsp; <sup>4</sup>Huazhong University of Science and Technology &nbsp; <sup>5</sup>University of the Chinese Academy of Sciences

<sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding authors

</div>

<p align="center">
  <img src="assets/case_study.png" width="90%" alt="PreCoMem on the coffee-to-tea example">
</p>
<p align="center"><em>Figure 1 of the paper. PreCoMem manages the lifecycle of facts and hypotheses and moves stably between conflicting belief states.</em></p>

## Overview

Long-horizon dialogue agents must keep user memories stable enough for personalization, yet adapt when the user genuinely changes. PreCoMem is a memory consolidation framework inspired by predictive coding. It represents memory as confidence-weighted beliefs about the user and revises them with one reliability-aware prediction-error signal.

For every turn, PreCoMem recalls the Top-K beliefs and computes an **effective surprisal** $S_{\mathrm{eff}} = \rho_t \cdot S_{\mathrm{raw}}$. The raw term mixes semantic deviation with an NLI contradiction score. The reliability coefficient $\rho_t$ shrinks it when the retrieval scores are diffuse. A two-threshold gate then routes the turn to one of three deterministic operators:

| Regime | Condition | Update |
|---|---|---|
| `MAINTAIN` | $S_{\mathrm{eff}} \le \theta_{\mathrm{low}}$ | reinforce supported beliefs: $w \leftarrow w + \eta\,(1-w)$ |
| `PROFILE` | $\theta_{\mathrm{low}} < S_{\mathrm{eff}} \le \theta_{\mathrm{high}}$ | write a low-confidence hypothesis (the "waiting room") |
| `CORRECT` | $S_{\mathrm{eff}} > \theta_{\mathrm{high}}$ | decay contradicted beliefs, $w \leftarrow w\, e^{-\beta S_{\mathrm{eff}} \zeta}$, and record the new observation |

Hypotheses are promoted to facts after repeated support and pruned when support never returns. The LLM sub-agents only decide *which* beliefs to touch and *how strongly*. Gating, reinforcement, decay, promotion and pruning are closed-form, so memory stays bounded and every update can be audited.

This repository contains:

- `precomem/`: the framework (Algorithm 1 of the paper) with typed configuration, per-turn traces and memory persistence.
- An evaluation harness for LoCoMo, LongMemEval-S, PersonaMem-v2 and DiMoBench that uses the QA and judge prompts of Appendix F verbatim.
- The six-stage generator–verifier construction pipeline of **DiMoBench** and **DiMoBench-XL**.
- A configuration switch for every ablation and control reported in the paper.
- Offline unit tests that need no API keys and no model downloads.

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/ZhongWishing/PreCoMem.git
cd PreCoMem
python -m venv .venv && source .venv/bin/activate
pip install -e ".[models,gemini]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `models` | `torch`, `transformers`, `sentence-transformers` | the embedder and the NLI verifier used in the paper |
| `gemini` | `google-genai` | the two auxiliary judges and the DiMoBench verifier |
| `calibrate` | `datasets` | fitting the NLI temperature on MNLI |
| `dev` | `pytest`, `ruff` | tests and linting |

`pip install -e ".[all]"` installs everything. Then set the keys of the providers you use:

```bash
export OPENAI_API_KEY=...    # backbone and primary judge
export GEMINI_API_KEY=...    # auxiliary judges and DiMoBench verifier
```

Without the `models` extra, PreCoMem falls back to a hashing embedder and a heuristic contradiction verifier. The pipeline then still runs for smoke tests, but it does not reproduce the paper's numbers. Set `PRECOMEM_STRICT_EMBEDDER=1` and `PRECOMEM_STRICT_NLI=1` to turn these fallbacks into errors.

## Models

PreCoMem is training-free and ships no weights of its own. It relies on two public checkpoints, which are downloaded automatically on first use:

- dense embeddings: [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- contradiction verifier: [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base)

The only fitted quantity is the scalar temperature $T$ of the verifier (Appendix C.2), fitted on the MNLI matched development split:

```bash
precomem-calibrate-nli --out configs/nli.yaml    # afterwards pass --config configs/nli.yaml
```

Backbones are called through the OpenAI Chat Completions API or any OpenAI-compatible server, for example vLLM serving a Qwen model (`configs/qwen_vllm.yaml`). Reasoning models that reject `max_tokens` or `temperature` are handled automatically. Extra request fields can be added with `--set backbone.extra_params.reasoning_effort=minimal`.

## Quick start

```python
from precomem import PreCoMem, load_config

agent = PreCoMem(load_config())
for utterance in [
    "I usually start my mornings with coffee.",
    "I tried matcha at a new cafe today, pretty refreshing.",
    "Coffee has been making my stomach uneasy lately.",
    "I really don't drink coffee anymore at all.",
]:
    trace = agent.step(utterance)   # retrieve, respond, surprisal, gate, operator, lifecycle
    print(trace.one_line())

print(agent.render_memory())        # beliefs with type, confidence and provenance
print(agent.query("What should I drink tomorrow morning?"))
agent.save_memory("memory.json")    # belief store, episodic cache and topic table
```

- `agent.observe(user, assistant, meta=...)` replays a recorded exchange without generating a reply. This is the benchmark setting.
- `agent.recall(question)` performs a read-only Top-K retrieval.
- Every `TurnTrace` records the retrieved evidence, the surprisal components, the gate decision, the closed-form updates, lifecycle counters, latency and token usage.

The same example runs from the command line with `precomem-demo`.

## Data

Place the benchmark files under `data/`, which git ignores.

| Benchmark | Source | Path used below |
|---|---|---|
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) (`data/locomo10.json`) | `data/locomo10.json` |
| LongMemEval-S | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) (500-question S split) | `data/longmemeval_s.json` |
| PersonaMem-v2 | [bowen-upenn/ImplicitPersona](https://huggingface.co/datasets/bowen-upenn/ImplicitPersona), linked from [bowen-upenn/PersonaMem](https://github.com/bowen-upenn/PersonaMem) | `data/personamem_v2_32k.json`, `data/personamem_v2_128k.json` |
| DiMoBench / -XL | built with `precomem-build-dimobench` ([see below](#dimobench-and-dimobench-xl)) | `data/dimobench/suite.json`, `data/dimobench_xl/suite.json` |

The LoCoMo and LongMemEval loaders read the official release formats directly. LongMemEval also publishes a cleaned S split (`longmemeval_s_cleaned.json`) with the same schema. For PersonaMem-v2, convert the release into the per-persona layout documented in [`precomem/benchmarks/personamem.py`](precomem/benchmarks/personamem.py), that is, one record per persona with its chat `history` and its `questions`.

## Reproducing the paper

All runs use three seeds `{13, 42, 1234}` and the judge protocol of Section 3.1. A primary `gpt-4.1-mini` judge and two `gemini-2.5-flash` judges with distinct system prompts score every answer, and accuracy is the mean of the three binary verdicts.

```bash
# Table 1, LoCoMo: gpt-4o-mini block (default) and gpt-4.1 block
precomem-run --benchmark locomo --data data/locomo10.json
precomem-run --benchmark locomo --data data/locomo10.json --backbone gpt-4.1

# Table 1, LongMemEval-S: gpt-4.1 block (default); Table 7: gpt-4o-mini
precomem-run --benchmark longmemeval --data data/longmemeval_s.json
precomem-run --benchmark longmemeval --data data/longmemeval_s.json --backbone gpt-4o-mini

# Figure 3(e)-(f), PersonaMem-v2 under gpt-4.1 (MCQ and Open-Ended questions)
precomem-run --benchmark personamem --data data/personamem_v2_32k.json
precomem-run --benchmark personamem --data data/personamem_v2_128k.json

# Section 3.3, DiMoBench and DiMoBench-XL
precomem-run --benchmark dimobench --data data/dimobench/suite.json
precomem-run --benchmark dimobench --data data/dimobench_xl/suite.json
```

| Flag | Purpose |
|---|---|
| `--backbone gpt-4.1`, `--base-url http://host:8000/v1` | choose the backbone; any OpenAI-compatible endpoint works |
| `--config configs/dynamic_threshold.yaml` | PreCoMem(DT), the adaptive-threshold variant (Appendix D.5) |
| `--set retrieval.top_k=5 --set gate.theta_high=0.7` | override any configuration field |
| `--seeds 13 --limit 20 --workers 8` | partial or faster runs |
| `--skip-judge` | inference only |

Each run writes `results/<benchmark>_<timestamp>/`:

| File | Content |
|---|---|
| `run_meta.json` | resolved configuration, models, package version and git commit |
| `seed_<n>/predictions.jsonl` | one record per question: prediction, judge votes, retrieved belief ids and weights, low-confidence share of the Top-K, prompt tokens |
| `seed_<n>/traces/<sample>.jsonl` | one record per replayed exchange (the Algorithm 1 trace) |
| `seed_<n>/memory/<sample>.json` | final belief store, episodic cache and topic table |
| `summary.json` | per-category accuracy as mean ± std over seeds, and response-model token usage |

Protocol details that matter for reproduction:

- The answer model receives exactly the memory input whose size the paper reports. This is the Top-K (K = 10) beliefs recalled for the question, at most eight linked episodic excerpts, and the six-turn working window. Confidence weights are never shown to it.
- LoCoMo uses the category numbering of the released data (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop) and excludes the adversarial category 5. This yields the 1,540 questions of Table 1.
- Each LongMemEval-S question has a question date. It is appended to the question, because temporal questions are relative to it.
- DiMoBench questions carry an `anchor_turn` and are asked during the replay, right after that turn. Their answers therefore reflect the belief state at that moment.
- Each replayed exchange makes one sub-agent call. Each question makes one answer call and three judge calls.

## Results

The numbers below are copied from the paper. They are LLM-judge accuracies in %, given as mean ± std over three seeds. *Avg. tokens* is the mean response-model input per query.

**LoCoMo** (Table 1, 1,540 questions)

| Backbone | Single-Hop | Multi-Hop | Temporal | Open-Domain | Overall | Avg. tokens |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| gpt-4o-mini | 81.41 ± 0.51 | 68.49 ± 0.98 | 66.21 ± 0.12 | 39.63 ± 0.25 | **73.57 ± 0.28** | 3.3k |
| gpt-4.1 | 90.41 ± 0.35 | 83.48 ± 0.29 | 84.74 ± 0.80 | 48.23 ± 0.65 | **85.06 ± 0.07** | 3.1k |

The strongest baseline, TiMem, reaches 71.88 (gpt-4o-mini) and 84.04 (gpt-4.1) overall.

**LongMemEval-S** (Table 1 for gpt-4.1, Table 7 for gpt-4o-mini, 500 questions)

| Backbone | SSU | SSA | SSP | MS | KU | TR | Overall | Avg. tokens |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| gpt-4o-mini | 91.58 ± 0.04 | 86.38 ± 0.19 | 79.40 ± 0.81 | 84.84 ± 0.22 | 87.13 ± 0.56 | 80.34 ± 0.73 | **84.28 ± 0.05** | 4.1k |
| gpt-4.1 | 94.09 ± 0.98 | 90.72 ± 0.12 | 83.93 ± 0.85 | 87.44 ± 0.99 | 91.80 ± 0.87 | 84.35 ± 0.83 | **89.39 ± 0.76** | 3.8k |

The strongest baseline, EverMemOS, reaches 82.32 (gpt-4o-mini) and 86.16 (gpt-4.1) overall.

**PersonaMem-v2** (Figure 3, gpt-4.1). Open-Ended accuracy is 72.31 at 32k context and 67.84 at 128k, which is 3.37 and 2.89 points above EverMemOS.

**DiMoBench** (Figure 4). After the change point, PreCoMem recovers fastest and stabilizes at 84%. EverMemOS, TiMem and O-Mem plateau at 78–80%.

The AMemGym and LoCoMo-Plus results of Appendix D.1 follow those benchmarks' official protocols and are not covered by the runners in this repository.

## Ablations and controls

Every ablation in the paper is a configuration switch, so a main run and its ablations differ only in flags.

```bash
precomem-run --benchmark locomo --data data/locomo10.json --set ablation.es=true
precomem-run --benchmark locomo --data data/locomo10.json --set surprisal.lambda=0
precomem-run --benchmark locomo --data data/locomo10.json --set "ablation.random_routing=[0.562,0.274,0.164]"
precomem-run --benchmark dimobench --data data/dimobench/suite.json --set ablation.without_profile=true
```

| Switch | Meaning |
|---|---|
| `ablation.es=true` | the gate score is forced to 0, so every evidence-bearing turn is `MAINTAIN` (reinforce-only lower bound) |
| `ablation.rr=true` | retrieval reliability $\rho_t = 1$ |
| `ablation.cw=true` | every belief weight is overwritten with 1 after each write: unweighted retrieval, immediate promotion, no pruning |
| `ablation.read_time_weighting=false` | drops $w_t(m)$ from the retrieval score only (factorial control) |
| `ablation.without_maintain=true`, `ablation.without_correct=true` | the regime fires but performs no update |
| `ablation.without_profile=true` | medium-surprisal observations are committed directly as facts, without the waiting room |
| `ablation.single_threshold=true` | the medium band is routed to `CORRECT` |
| `ablation.random_routing=[pM,pP,pC]` | frequency-matched random routing with the read path unchanged |
| `surprisal.lambda=0` or `=1` | removes $d_{\mathrm{sem}}$ or $d_{\mathrm{conf}}$ from $S_{\mathrm{raw}}$ |
| `lifecycle.fact_optional_prune=0.05` | hard-prunes low-confidence facts (fact-pruning ablation) |
| `gate.adaptive=true` | PreCoMem(DT) |

## DiMoBench and DiMoBench-XL

DiMoBench isolates the *write path* under gradual, implicit belief drift with adversarial retrieval interference. `precomem/dimobench/` implements the six-stage construction pipeline of Appendix D.6 with a strict separation between the generator and the verifier.

| Stage | What | Model |
|:-:|---|---|
| 1 | persona card: Big-Five profile, life stage, confusable social circle, drift triggers | generator |
| 2 | life skeleton: session summaries and *cue scripts* that say which implicit event expresses which slot state | generator |
| 3 | dialogue: 10-turn sessions written in order under a rolling summary, with every cue localized to a turn | generator |
| 4 | questions: one per (reasoning category × interference tag) cell of a fixed quota grid, with evidence pointers `D{session}:{turn}` | generator |
| 5 | verification on six criteria; `REVISE` regenerates with feedback (up to 3 rounds) and `REJECT` discards | verifier |
| 6 | 20% stratified audit shard, annotation sheets and a Cohen's κ helper | offline |

`precomem/dimobench/spec.py` fixes the structure, so the LLM stages only supply language. The base track has 35 sessions of 10 turns and a drift at turn 200. Four implicit events in session 21 introduce the drift, sessions 22–24 carry dense cues, and sessions 25–35 form a maintenance tail. Confounders appear every 30 turns, and the 200 questions split into 90 pre-change, 10 change-point and 100 post-change. The XL track has 105 sessions, five attribute slots and three drifts at turns 280, 580 and 870. Each drift is expressed through 22 cues, and every 50-turn window receives 20 questions (420 per persona).

```bash
precomem-build-dimobench --variant base --out data/dimobench        # GPT-4o generator, Gemini-2.5-Pro verifier
precomem-build-dimobench --variant xl   --out data/dimobench_xl
precomem-build-dimobench --variant base --personas persona_a --max-sessions 4 --max-qa 10   # cheap smoke run
```

Persona seeds live in [`precomem/configs/dimobench_personas.yaml`](precomem/configs/dimobench_personas.yaml). Builds checkpoint every stage under `<out>/_work/` and resume automatically. Each persona file contains the dialogue, the gold slot-state timeline, every planned cue with the turn that realizes it, the questions with their verifier verdicts, and build statistics. The `audit/` folder holds the annotation sheets. Generation is stochastic, so a rebuilt suite differs in wording from the dialogues used in the paper.

## Configuration

All hyperparameters live in [`precomem/configs/default.yaml`](precomem/configs/default.yaml) and are parsed into typed, validated dataclasses (`precomem/config.py`). Unknown keys and violated invariants, such as a hypothesis band that crosses $\tau_{\mathrm{promote}}$, fail at load time.

| YAML key | Paper symbol |
|---|---|
| `retrieval.top_k`, `retrieval.alpha.{semantic,bm25,topic}` | $K$ and $(\alpha_1, \alpha_2, \alpha_3)$ in Eq. 1 |
| `surprisal.lambda`, `surprisal.reliability_eps` | $\lambda$ in Eq. 4 and $\epsilon$ in Eq. 3 |
| `gate.theta_low`, `gate.theta_high` | $(\theta_{\mathrm{low}}, \theta_{\mathrm{high}}) = (0.2, 0.6)$ in Eq. 6 |
| `gate.adaptive*` | PreCoMem(DT): window, weights, threshold ranges and smoothing |
| `operators.maintain.eta_{min,max}` | clip range of $\eta_t(m)$ in Eq. 7 |
| `operators.profile.{explicit,latent}_band` | the class bands $[w_{\min}, w_{\max}]$ in Eq. 8 |
| `operators.correct.beta`, `operators.correct.new_fact_weight` | $\beta$ in Eq. 10, and the weight of the fresh belief node |
| `lifecycle.decay_gamma`, `lifecycle.promote_threshold`, `lifecycle.hypothesis_prune` | $\gamma$ in Eq. 9, $\tau_{\mathrm{promote}} = 0.80$ and the prune threshold 0.10 |
| `retrieval.short_term_window`, `retrieval.max_provenance_traces` | the six-turn window and the eight-excerpt cap of the response input |
| `nli.temperature` | the temperature $T$ of the calibrated verifier |
| `backbone.extra_params` | extra fields merged into every backbone request |
| `ablation.*` | the switches listed above |

## Repository structure

```
precomem/
  agent.py          PreCoMem: Algorithm 1 (observe / step / recall / query, traces, persistence)
  config.py         typed configuration, YAML loading and validation
  gate.py           two-threshold gate with fixed and adaptive (DT) threshold policies
  trace.py          per-turn trace records
  memory/           MemoryItem, EpisodicTrace, BeliefStore, EpisodicCache, TopicTable, MemoryState
  retrieval/        confidence-modulated tri-signal retriever, BM25 and embedders
  surprisal/        rho_t, d_sem, d_conf, S_raw and S_eff, plus the calibrated NLI verifier
  operators/        MAINTAIN, PROFILE, CORRECT and the hypothesis lifecycle
  llm/              provider clients (OpenAI-compatible and Gemini) and the sub-agent prompts
  benchmarks/       loaders for LoCoMo, LongMemEval-S, PersonaMem-v2 and DiMoBench
  evaluation/       QA and judge prompts, judge ensemble, metrics and the benchmark runner
  dimobench/        construction pipeline (spec, schema, prompts, pipeline, audit)
  cli/              precomem-run, precomem-demo, precomem-build-dimobench, precomem-calibrate-nli
  configs/          default.yaml and dimobench_personas.yaml
configs/            example override files (gpt-4.1, PreCoMem(DT), vLLM backbone, offline smoke test)
tests/              offline unit tests
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite runs offline in a few seconds. It checks the closed-form updates against the worked examples of the paper, such as the `PROFILE` initialization $w_0 = 0.6549$ (Appendix C.3) and the eight-turn no-support horizon of an unsupported latent hypothesis. It also covers the gate, the lifecycle, retrieval, every ablation switch, the benchmark loaders, the runner with anchored questions, the provider clients and the complete DiMoBench pipeline with scripted generator and verifier doubles.

## License

The code is released under the [MIT License](LICENSE). The benchmarks used for evaluation keep their original licenses.

## Contact

Please open a [GitHub issue](https://github.com/ZhongWishing/PreCoMem/issues) for questions and bug reports, or contact Jian Zhong (zhongjian@mail.nankai.edu.cn).
