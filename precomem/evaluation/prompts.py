"""QA and LLM-as-judge prompts, verbatim from Appendix F ("Dataset Evaluation Prompts").

The QA prompts receive the rendered memory context (Top-K beliefs plus linked
episodic excerpts) and the question; the judge prompts receive the question,
the gold answer and the generated answer and return ``{"label": ...}``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# QA prompts
# ---------------------------------------------------------------------------

LOCOMO_QA_PROMPT = """\
You are an intelligent memory assistant tasked with retrieving accurate information from
conversation memories.

# CONTEXT:
You have access to relevant memories extracted from a conversation history.
- Input Format: List of ordered dialogue turns containing a Dialogue ID (e.g., D1:3), Speaker
Name, Timestamp (e.g., "1:56 pm on 8 May, 2023"), and Text content.
- Task: Use these memories to answer the user's question accurately.

# INSTRUCTIONS:
1. Analyze: Examine all provided memories carefully. Pay strict attention to the Timestamp
associated with each specific dialogue turn.
2. Temporal Reasoning:
   - You MUST convert all relative time references (e.g., "last month", "three days later") into
     absolute dates (Day, Month, Year).
   - Calculate these dates relative to the timestamp of the specific memory where the event is
     mentioned.
   - Example: If a memory dated "10 Oct 2023" says "I went there last month", the real date is
     "September 2023".
3. Conflict Resolution: If memories contain contradictory information, prioritize the
   information from the most recent timestamp.
4. Format Constraints:
   - Output the final answer concisely.
   - Use Arabic digits for any counts, quantities, or years (e.g., "5" instead of "five").
   - If the answer is not explicitly found or deducible, output exactly: Unknown.

# APPROACH (Think Step-by-Step):
1. Locate the dialogue turns relevant to the question.
2. Check the exact timestamp of those turns.
3. Perform date calculations if relative time is mentioned.
4. Check for any later updates that might contradict earlier facts.
5. Formulate the final answer.

# INPUT DATA:
Relevant Memories:
{context_memories}

Question: {question}

# OUTPUT FORMAT:
Thought: [Briefly explain your reasoning and date calculations here]
Answer: [Your final concise answer here]
"""


LONGMEMEVAL_QA_PROMPT = """\
You are an intelligent memory assistant tasked with answering questions about a user based on
retrieved memory items extracted from prior multi-session chat interactions.

# CONTEXT:
LongMemEval-S evaluates six question types (Single-Session-User, Single-Session-Assistant,
Single-Session-Preference, Multi-Session, Knowledge-Update, Temporal-Reasoning), each
requiring different reasoning over user information accumulated across sessions. Some questions
are intentionally unanswerable: in that case the correct response is to abstain.
- Input Format: Each retrieved item carries its session/turn identifier, an absolute timestamp
  (when available), the speaker (User or Assistant), and the text content.
- Task: Use these memories to answer the user's question accurately.

# INSTRUCTIONS:
1. Locate the relevant evidence: For single-session questions (SSU/SSA/SSP), the answer is
   grounded in one session; for multi-session (MS) questions, you must aggregate or compare
   across two or more sessions.
2. Knowledge-update reasoning (KU): If two memories conflict (e.g., the user moves city,
   changes job, switches diet), trust the most recent timestamped statement and treat earlier
   statements as outdated.
3. Temporal reasoning (TR):
   - Convert relative time references (e.g., "last month", "three days ago") into absolute
     dates computed from the timestamp of the memory in which the reference appears.
   - Use the metadata timestamps (not the response time) when comparing event order.
4. Speaker grounding: If the question asks about a user preference or assistant suggestion,
   ensure the cited memory was authored by the correct speaker (User vs. Assistant).
5. Abstention: If the retrieved memories do not contain enough information to answer the
   question, output exactly: I don't know. Do not guess or fabricate.
6. Format Constraints:
   - Output the final answer concisely.
   - Use Arabic digits for any counts, quantities, or years (e.g., "5" instead of "five").

# APPROACH (Think Step-by-Step):
1. Identify the question type and the speaker whose information is queried.
2. Locate the relevant memory items and check their timestamps.
3. Resolve any knowledge updates by preferring the most recent statement.
4. Perform any necessary temporal arithmetic.
5. If sufficient evidence is present, formulate the final answer; otherwise abstain.

# INPUT DATA:
Relevant Memories:
{context_memories}

Question: {question}

# OUTPUT FORMAT:
Thought: [Briefly explain which memories you used, any conflict resolution, and any date
calculations]
Answer: [Your final concise answer, or "I don't know"]
"""


PERSONAMEM_QA_PROMPT = """\
You are an intelligent assistant capable of synthesizing retrieved memory fragments to answer
questions about a user's persona.

# TASK:
Answer the current user question by selecting the best option from the provided list. You must
rely STRICTLY on the provided "Retrieved Memories".

# INPUT DATA:
1. *Retrieved Memories:* A collection of memory snippets retrieved from the user's past
interactions. These may be fragmented or out of strict order.
2. *Current Question:* The user's specific query or message.
3. *Options:* The candidate answers.

# CRITICAL INSTRUCTIONS:
1. *Synthesize and Verify:*
- The retrieved memories might contain conflicting information (e.g., a memory from last year
  saying "I eat meat" vs. a memory from yesterday saying "I am vegetarian now").
- You must strictly follow the *timeline* of the interactions.
- If two memories conflict, the *MOST RECENT* information (latest timestamp or latest
  interaction ID) trumps older information.
- Prioritize the information that logically represents the *current* state of the user.
2. *Distinguish Speaker:*
- Ensure the preference belongs to the **User**, not the Assistant suggesting something.
3. *Handle Missing Info:*
- If the retrieved memories do not contain the answer, select the option representing
  "Unknown" or the most neutral choice. Do not guess.
4. *Output Format:*
- *Thought:* Briefly analyze the retrieved memories. Resolve any conflicts if present.
- *Answer:* Output ONLY the option label (e.g. "A", "B", "C", "D" etc.).

# DATA:
Retrieved Memories : {retrieved_memories}
Current Question   : {user_question_or_message}
Option             : {all_options}

# RESPONSE:
Thought: [Analyze memories -> Resolve conflicts -> Match option]
Answer: [Option Label]
"""


DIMOBENCH_QA_PROMPT = """\
You are an intelligent memory assistant tasked with retrieving accurate information from
conversation memories.

# CONTEXT:
You have access to relevant memories extracted from a conversation history.
- Input Format: List of ordered dialogue turns containing a Dialogue ID (e.g., D10:6), Speaker
  Name, Timestamp or Turn Index, and Text content.
- Task: Use these memories to answer the user's question accurately.

# INSTRUCTIONS:
1. Analyze: Examine all provided memories carefully. Pay strict attention to the Timestamp or
   Turn Order associated with each specific dialogue turn.
2. Temporal Reasoning:
   - If calendar timestamps are provided, you MUST convert all relative time references (e.g.,
     "last month", "three days later") into absolute dates (Day, Month, Year).
   - Calculate these dates relative to the timestamp of the specific memory where the event is
     mentioned.
   - Example: If a memory dated "10 Oct 2023" says "I went there last month", the real date is
     "September 2023".
   - If only turn indices are provided (no calendar dates), use turn order to resolve
     "before/after" and do NOT invent dates.
3. Conflict Resolution: If memories contain contradictory information, prioritize the
   information from the most recent timestamp or latest turn.
4. Format Constraints:
   - Output the final answer concisely.
   - Use Arabic digits for any counts, quantities, or years (e.g., "5" instead of "five").
   - If the answer is not explicitly found or deducible, output exactly: Unknown.

# APPROACH (Think Step-by-Step):
1. Locate the dialogue turns relevant to the question.
2. Check the exact timestamp or turn order of those turns.
3. Perform date calculations if relative time is mentioned.
4. Check for any later updates that might contradict earlier facts.
5. Formulate the final answer.

# INPUT DATA:
Relevant Memories:
{context_memories}

Question: {question}

# OUTPUT FORMAT:
Thought: [Briefly explain your reasoning and date calculations here]
Answer: [Your final concise answer here]
"""


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

LOCOMO_JUDGE_PROMPT = """\
Your task is to label an answer to a question as "CORRECT" or "WRONG".
You will be given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

The question asks about something one user should know about the other user based on their
prior conversations.
The gold answer is usually concise. The generated answer may be longer.
Be generous: if the generated answer clearly refers to the same fact/topic as the gold answer,
mark it CORRECT.
For time-related questions, accept equivalent dates/periods even if the format differs (e.g.,
"May 7th" vs "7 May").
If the gold answer is "Unknown", only "Unknown" (or a clear abstention) is CORRECT.

Now judge the real case.

Question: {question}
Gold answer: {standard_answer}
Generated answer: {generated_answer}

Return JSON only:
{{"label": "CORRECT" | "WRONG"}}
"""


LONGMEMEVAL_JUDGE_PROMPT = """\
Your task is to label an answer to a question as "CORRECT" or "WRONG".
You will be given:
(1) a question (with its question type tag, e.g., SSU/SSA/SSP/MS/KU/TR),
(2) a gold (ground-truth) answer,
(3) a generated answer.

The question concerns long-term user information accumulated across multiple chat sessions.
The gold answer is usually concise; the generated answer may be longer or include reasoning.

Apply the following rules consistently with the question type:
- Information extraction (SSU/SSA): Mark CORRECT if the generated answer clearly refers to
  the same user fact or assistant statement as the gold answer; minor paraphrasing is
  acceptable.
- Single-session preference (SSP): Mark CORRECT if the generated response reflects the same
  preference or recommendation rationale as the gold answer; surface form may differ.
- Multi-session reasoning (MS): Mark CORRECT if the generated answer correctly aggregates or
  compares the relevant facts; partial overlap that misses the comparison is WRONG.
- Knowledge update (KU): Mark CORRECT only if the generated answer reflects the most recent
  state of the user; answers based on outdated information are WRONG.
- Temporal reasoning (TR): Accept equivalent dates or periods even if the format differs
  (e.g., "May 7th" vs. "7 May"; "last September" vs. "September 2023").
- Abstention: If the gold answer is "I don't know" / "Unknown", only an explicit abstention
  or refusal in the generated answer is CORRECT; any concrete guess is WRONG.

Now judge the real case.

Question: {question}
Question type: {question_type}
Gold answer: {standard_answer}
Generated answer: {generated_answer}

Return JSON only:
{{"label": "CORRECT" | "WRONG"}}
"""


PERSONAMEM_JUDGE_PROMPT = """\
Your task is to label an answer to a multiple-choice question as "CORRECT" or "WRONG".
You will be given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer,
(4) the list of options.

The gold answer is usually an option label like "(c)".
The generated answer may be a label or the full text of an option.
Be generous: if the generated answer matches the gold label, or matches the gold option text,
mark it CORRECT.
If the gold answer is "Unknown", only "Unknown" (or a clear abstention) is CORRECT.

Now judge the real case.

Question: {question}
Gold answer: {standard_answer}
Generated answer: {generated_answer}
Options:
{all_options}

Return JSON only:
{{"label": "CORRECT" | "WRONG"}}
"""


DIMOBENCH_JUDGE_PROMPT = """\
Your task is to label an answer to a question as "CORRECT" or "WRONG".
You will be given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

The question asks about a user's state, history, or change-point based on prior dialogue.
Be generous: if the generated answer clearly refers to the same fact/state/turn as the gold
answer, mark it CORRECT.
For time/turn-related questions, accept equivalent expressions that point to the same time
period or turn.
If the gold answer is "Unknown", only "Unknown" (or a clear abstention) is CORRECT.

Now judge the real case.

Question: {question}
Gold answer: {standard_answer}
Generated answer: {generated_answer}

Return JSON only:
{{"label": "CORRECT" | "WRONG"}}
"""


# ---------------------------------------------------------------------------
# PersonaMem-v2 Open-Ended format
# ---------------------------------------------------------------------------
# Appendix F lists the multiple-choice PersonaMem-v2 prompts. For the
# Open-Ended format the same instructions are used with the option list
# removed and a free-form answer requested; the judge then checks the answer
# against the reference response instead of an option label.

PERSONAMEM_OPEN_QA_PROMPT = """\
You are an intelligent assistant capable of synthesizing retrieved memory fragments to answer
questions about a user's persona.

# TASK:
Respond to the current user question or message in a personalized way. You must rely STRICTLY
on the provided "Retrieved Memories".

# INPUT DATA:
1. *Retrieved Memories:* A collection of memory snippets retrieved from the user's past
interactions. These may be fragmented or out of strict order.
2. *Current Question:* The user's specific query or message.

# CRITICAL INSTRUCTIONS:
1. *Synthesize and Verify:*
- The retrieved memories might contain conflicting information (e.g., a memory from last year
  saying "I eat meat" vs. a memory from yesterday saying "I am vegetarian now").
- You must strictly follow the *timeline* of the interactions.
- If two memories conflict, the *MOST RECENT* information (latest timestamp or latest
  interaction ID) trumps older information.
- Prioritize the information that logically represents the *current* state of the user.
2. *Distinguish Speaker:*
- Ensure the preference belongs to the **User**, not the Assistant suggesting something.
3. *Handle Missing Info:*
- If the retrieved memories do not contain the relevant preference, give a helpful but neutral
  response. Do not guess.
4. *Output Format:*
- *Thought:* Briefly analyze the retrieved memories. Resolve any conflicts if present.
- *Answer:* A concise response (one to three sentences) tailored to the user's current state.

# DATA:
Retrieved Memories : {retrieved_memories}
Current Question   : {user_question_or_message}

# RESPONSE:
Thought: [Analyze memories -> Resolve conflicts -> Tailor the response]
Answer: [Your response]
"""


PERSONAMEM_OPEN_JUDGE_PROMPT = """\
Your task is to label a response to a user's message as "CORRECT" or "WRONG".
You will be given:
(1) the user's question or message,
(2) a gold (ground truth) answer describing the response that fits the user's current preferences,
(3) a generated response.

Mark the generated response CORRECT if it is consistent with the preference expressed in the gold
answer. Mark it WRONG if it contradicts that preference, relies on an outdated preference or on a
preference that belongs to someone else, or ignores the preference entirely.
If the gold answer is "Unknown", only "Unknown" (or a clear abstention) is CORRECT.

Now judge the real case.

Question: {question}
Gold answer: {standard_answer}
Generated answer: {generated_answer}

Return JSON only:
{{"label": "CORRECT" | "WRONG"}}
"""


EVAL_PROMPTS: Dict[str, str] = {
    "locomo": LOCOMO_QA_PROMPT,
    "longmemeval": LONGMEMEVAL_QA_PROMPT,
    "personamem": PERSONAMEM_QA_PROMPT,
    "dimobench": DIMOBENCH_QA_PROMPT,
}


JUDGE_PROMPTS: Dict[str, str] = {
    "locomo": LOCOMO_JUDGE_PROMPT,
    "longmemeval": LONGMEMEVAL_JUDGE_PROMPT,
    "personamem": PERSONAMEM_JUDGE_PROMPT,
    "dimobench": DIMOBENCH_JUDGE_PROMPT,
}


def build_qa_prompt(
    benchmark: str,
    question: str,
    context_memories: str,
    options: Optional[List[str]] = None,
    question_date: Optional[str] = None,
) -> str:
    """Fill the benchmark-specific QA prompt.

    PersonaMem-v2 questions without options use the Open-Ended variant.
    ``question_date`` (LongMemEval-S) is the date the question is asked on;
    temporal-reasoning questions such as "how many days ago" are relative to
    it, so it is appended to the question.
    """
    key = benchmark.lower()
    if key not in EVAL_PROMPTS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if question_date:
        question = f"{question}\nCurrent date: {question_date}"
    if key == "personamem":
        if not options:
            return PERSONAMEM_OPEN_QA_PROMPT.format(
                retrieved_memories=context_memories, user_question_or_message=question
            )
        return PERSONAMEM_QA_PROMPT.format(
            retrieved_memories=context_memories,
            user_question_or_message=question,
            all_options="\n".join(options),
        )
    return EVAL_PROMPTS[key].format(context_memories=context_memories, question=question)


def build_judge_prompt(
    benchmark: str,
    question: str,
    gold: str,
    prediction: str,
    question_type: Optional[str] = None,
    options: Optional[List[str]] = None,
) -> str:
    """Fill the benchmark-specific judge prompt."""
    key = benchmark.lower()
    if key not in JUDGE_PROMPTS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    template = JUDGE_PROMPTS[key]
    if key == "longmemeval":
        return template.format(
            question=question,
            question_type=question_type or "UNKNOWN",
            standard_answer=gold,
            generated_answer=prediction,
        )
    if key == "personamem":
        if not options:
            return PERSONAMEM_OPEN_JUDGE_PROMPT.format(
                question=question, standard_answer=gold, generated_answer=prediction
            )
        return template.format(
            question=question,
            standard_answer=gold,
            generated_answer=prediction,
            all_options="\n".join(options),
        )
    return template.format(question=question, standard_answer=gold, generated_answer=prediction)
