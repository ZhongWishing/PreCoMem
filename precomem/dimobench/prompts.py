"""Generator and verifier prompts for the DiMoBench construction pipeline.

Every prompt opens with a ``[DIMOBENCH STAGE: ...]`` marker and embeds the
machine-readable plan it was built from under ``PLAN_JSON``. The marker makes
logs and routing (e.g. test doubles) unambiguous; the plan block is what lets
the model reference cue ids and dialogue ids exactly.
"""

from __future__ import annotations

PERSONA_PROMPT = """\
[DIMOBENCH STAGE: persona]
You are designing a synthetic user persona for DiMoBench, a benchmark about
how a conversational assistant's long-term memory should track *gradual,
implicit* changes in a user's attitudes and life circumstances.

Seed specification (authoritative; keep every slot name and every state
string exactly as given):
PLAN_JSON
{plan_json}

Design requirements:
- The persona must be psychologically plausible and consistent with the
  Big-Five profile and life stage in the seed.
- Give the persona a first name and a speaking style (register, typical
  sentence length, verbal habits) that will stay recognisable over hundreds
  of turns.
- Produce a social circle of 3 named characters. At least two names must be
  deliberately confusable (e.g. "Leo" and "Leon", "Mia" and "Mila"); state
  explicitly which pair is confusable. Each character has a one-phrase
  relation to the user.
- Produce 4-6 routine activities that fit the *initial* states of the slots.
- For every drift in the seed, write one concrete, event-driven trigger (a
  specific thing that happens in the persona's life) that makes the planned
  state change plausible without anyone ever naming the change.

Return strict JSON:
{{
  "name": "...",
  "summary": "2-3 sentences, third person",
  "speaking_style": "...",
  "social_circle": [
    {{"name": "...", "relation": "...", "confusable_with": "<name or null>"}}
  ],
  "routine": ["...", "..."],
  "drift_triggers": {{"1": "...", "2": "...", "3": "..."}}
}}
Include only the drift indices present in the seed. Return JSON only.
"""


SKELETON_PROMPT = """\
[DIMOBENCH STAGE: skeleton]
You are writing the life skeleton for sessions {first_session}-{last_session} of a
long DiMoBench dialogue between the user and an assistant ({turns_per_session}
turns per session, {n_sessions} sessions in total, {n_turns} turns).

Persona:
{persona_json}

Gold state of every attribute slot across the whole dialogue (turn ranges):
{timeline_json}

Drift triggers (never stated explicitly in the dialogue): {triggers_json}

Summary of the sessions written so far: {previous_summary}

Session plan for this chunk. For every session you get its phase, the cue
ids it must express, and its confounder directive:
PLAN_JSON
{plan_json}

Phase semantics:
- "pre": stable baseline; cues show the *current* state of their slot through
  routine, plans, reactions and small decisions.
- "change-point": the drift trigger has just happened. Cues are implicit
  events (a declined invitation, a changed routine, a hedge) that hint at the
  new state without anyone naming a change. Keep them deniable in isolation.
- "post-change-dense": the new state is now visible through denser,
  consistent behavior. No meta commentary ("I've changed").
- "post-change-maintenance": low frequency; an occasional cue confirms the
  new state as the new normal. Divergence from the post-drift baseline must
  shrink monotonically over the maintenance stage.

Confounder directives:
- "keyword_trap": the session contains irrelevant chit-chat (often about
  someone else, a show, a news item) that shares surface keywords with the
  focus slot but carries no information about the user's own state.
- "hypothetical": the user narrates something hypothetical, counterfactual
  or imagined ("if I ever...", "imagine I...") about the focus slot; it is
  not a fact and must be written so that a careful reader can tell.
- When "mention_confusable_entity" is true, one of the confusable characters
  does or says something that a careless reader could attribute to the user.

For each session write: a one-sentence summary, a time/place setting with an
explicit temporal marker relative to the previous session, and for every cue
id a concrete behavioral description plus a short surface hint. Cue
descriptions must express "state_to_express" implicitly; never use the slot
name or sentiment words such as like/love/hate/prefer about the slot itself.

Return strict JSON:
{{
  "sessions": [
    {{
      "index": <int>,
      "summary": "...",
      "setting": "...",
      "cues": [{{"cue_id": "...", "description": "...", "surface_hint": "..."}}],
      "confounder_script": "<what the confounder is about, or empty string>"
    }}
  ],
  "chunk_summary": "3-5 sentences summarizing these sessions for the next chunk"
}}
Return JSON only.
"""


DIALOGUE_PROMPT = """\
[DIMOBENCH STAGE: dialogue]
You are writing Session {session_index} of a DiMoBench dialogue between the
user ({user_name}) and an assistant. Produce exactly {turns_per_session}
turns, alternating user / assistant and starting with the user.

Persona:
{persona_json}

Rolling summary of the conversation so far:
{rolling_summary}

Last turns of the previous session (for continuity):
{previous_tail}

Session plan:
PLAN_JSON
{plan_json}

Rules:
1. Every cue in the plan must be realized in a USER turn, implicitly: through
   an activity choice, a reaction to an invitation, a plan, a small decision,
   or hedging language. Never state the slot name, never use explicit
   sentiment words about the slot (like/love/hate/prefer/enjoy), and never
   add meta commentary ("I've changed", "I'm different now").
2. Use the character names from the persona exactly; when the plan asks for
   a confusable-entity mention, make a confusable character the agent of an
   action that superficially resembles the user's own behavior.
3. If the plan has a confounder, realize it in one or two turns as described
   ("keyword_trap" = irrelevant chit-chat sharing keywords; "hypothetical" =
   clearly imagined / counterfactual narration, marked with words such as
   "if", "imagine", "suppose", "would").
4. Put an explicit temporal marker ("last weekend", "since Tuesday", "this
   month") in roughly one third of the user turns; the first user turn must
   anchor the session in time relative to the previous one.
5. Keep each turn under 45 words; the assistant is helpful and brief and
   never analyses the user's personality.
6. Stay in the persona's speaking style. Natural small talk and topic shifts
   are welcome; the cues should feel incidental, not announced.

Return strict JSON:
{{
  "turns": [
    {{"speaker": "user", "text": "..."}},
    {{"speaker": "assistant", "text": "..."}}
  ],
  "cue_realizations": [{{"cue_id": "...", "turn": <1-based index within this session>}}],
  "confounder_turns": [<1-based turn indices that carry the confounder, if any>],
  "entity_decoy_turns": [<1-based turn indices where a confusable character acts, if any>],
  "rolling_summary": "<updated 4-6 sentence summary of the whole conversation so far>"
}}
Return JSON only.
"""


QA_PROMPT = """\
[DIMOBENCH STAGE: qa]
You are writing evaluation questions for a DiMoBench dialogue. Each question
is asked right after dialogue turn {anchor_turn} (session {anchor_session});
the system answering it has seen turns 1-{anchor_turn} and nothing later.

Persona:
{persona_json}

Gold slot states valid at turn {anchor_turn}:
{states_json}
{previous_states_block}
Dialogue excerpts available as evidence (id, speaker, text). Only these ids
may be cited:
{excerpts}

Question requests. Each request fixes the reasoning category, the
interference tag and the focus slot; write exactly one question per request:
PLAN_JSON
{plan_json}

Category definitions:
- "current-state inference": what is true of the user's focus slot *now*
  (at turn {anchor_turn}), inferred from behavior.
- "history recall": a specific earlier fact, event or routine, cited at its
  original turn(s).
- "change-point detection": whether, when, or through which behaviors the
  user's focus slot shifted; before a drift the correct answer is that no
  shift has happened.
- "hypothesis-vs-fact distinction": separate what the user actually did or
  said from what was hypothetical, speculative or attributable to someone
  else.

Interference definitions (the question must be designed around the tag):
- "entity_confusable": the question tempts confusion between the user and a
  confusable character, or between two confusable characters; cite the
  decoy turn as well as the true evidence.
- "keyword_trap": a cited or nearby turn shares keywords with the slot but is
  irrelevant chit-chat; the answer must ignore it.
- "temporal_decoy": the question quotes or alludes to an earlier statement
  that may be outdated; the answer must reflect the state at turn {anchor_turn}
  (which may or may not have changed).
- "hypothetical_vs_fact": the question tempts treating a hypothetical
  narration as fact.
- "none": no designed interference.

Writing rules:
- Questions ask about behavior, plans, actions and reactions; do not use the
  slot name or sentiment words (like/love/hate/prefer/enjoy) about the slot.
  Refer to the user by name.
- Answers are concise (one sentence), unambiguous, and fully supported by
  the cited evidence; for questions whose correct answer is that nothing
  changed, say so plainly.
- "evidence" lists every id needed to justify the answer (1-4 ids, all from
  the excerpts above); "distractor_ids" lists the decoy ids the question is
  built around (may be empty).

Return strict JSON:
{{
  "questions": [
    {{
      "qid": "<copy from the request>",
      "question": "...",
      "answer": "...",
      "evidence": ["D7:5", "D13:9"],
      "distractor_ids": ["D12:3"],
      "rationale": "one sentence on why the evidence settles the answer"
    }}
  ]
}}
Return JSON only.
"""


REVISE_SUFFIX = """

REVISION REQUEST for {qid}: an independent verifier returned "{decision}" with
this feedback:
{feedback}
Rewrite only this question (same category, tag, focus slot and anchor) so
that the feedback is addressed. Return the same JSON schema with one entry.
"""


VERIFIER_PROMPT = """\
[DIMOBENCH STAGE: verify]
You are an independent verifier for one DiMoBench question. You did not write
the dialogue or the question; judge strictly.

Persona summary: {persona_summary}

Gold slot states valid at turn {anchor_turn}:
{states_json}
{previous_states_block}
Requirements the question was supposed to satisfy:
PLAN_JSON
{plan_json}

Dialogue excerpts: the cited evidence, the cited distractors, and one turn of
context on each side (id, speaker, text):
{excerpts}

The question:
{qa_json}

Score six criteria with 0 or 1:
1. answer_grounding     - the answer is fully supported by the cited evidence.
2. evidence_complete    - every turn needed for the answer is cited; nothing
                          essential is missing.
3. temporal_consistency - the answer reflects the gold state at turn {anchor_turn},
                          not an earlier or later state.
4. entity_accuracy      - every person referenced is the right one; decoys are
                          not misattributed to the user.
5. interference_valid   - the question genuinely instantiates its interference
                          tag ("none" is valid when no decoy is designed in).
6. drift_coherent       - the behavior the question relies on is consistent
                          with a gradual, plausible change (or stable state) of
                          the persona; nothing in the cited turns announces a
                          change explicitly.

Decision rule: ACCEPT if all six are 1; REJECT if answer_grounding or
temporal_consistency is 0 and the question cannot be fixed by rewording;
otherwise REVISE with concrete, actionable feedback.

Return strict JSON:
{{
  "scores": {{"answer_grounding": 0|1, "evidence_complete": 0|1, "temporal_consistency": 0|1,
             "entity_accuracy": 0|1, "interference_valid": 0|1, "drift_coherent": 0|1}},
  "decision": "ACCEPT" | "REVISE" | "REJECT",
  "feedback": "..."
}}
Return JSON only.
"""
