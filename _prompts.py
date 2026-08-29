"""Cell prompt builders for paradigm 2 (LLM-as-judge OAT factorial).

Each spec cell (C1F1D1 ... C2F2D2) produces a (system_prompt, user_prompt) pair.
The system prompt encodes C and F; the user prompt encodes D plus the (question,
answer_a, answer_b) triple via Arena-Hard's prompt_template.

C1 = OG_ARENA_HARD_PROMPT (Arena-Hard v2.0 shipped default; reasoning + 5-point scale)
C2 = Multi-criteria with explicit priority (correctness primary, helpfulness/safety constraints)
F1 = Reasoning before verdict (built into anchor)
F2 = Label-only verdict (5-point scale retained, reasoning instructions stripped)
D1 = No domain context
D2 = Coding-assistance scope preamble
"""

# --- C1 = OG_ARENA_HARD_PROMPT (with reasoning, 5-point scale) ---
OG_ARENA_HARD_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: \"My final verdict is tie: [[A=B]]\"."""

# --- C1 with F2 (label-only): same rubric criteria, no reasoning instructions ---
C1_LABEL_ONLY = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Consider whether each assistant's answer is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Also consider creativity and novelty when relevant, and identify any missing important information.

Output only one of the following choices as your final verdict with a label, with no explanation:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: \"[[A=B]]\"."""

# --- C2 = Multi-criteria priority rubric (with reasoning, F1) ---
C2_PRIORITY_REASONED = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer.

Evaluate the responses on three criteria with explicit priority:
- CORRECTNESS (primary): Are the technical claims, code, calculations, and reasoning accurate? Is the answer factually right?
- HELPFULNESS (constraint): Does the answer address the user's actual request? Is it well-organised and easy to follow?
- SAFETY (constraint): Does the answer avoid harmful, deceptive, or unethical content?

In your final verdict, prioritise correctness. A response that is more helpful but less correct should be rated lower than a response that is correct but less helpful. Helpfulness and safety are constraints, not primary criteria.

Begin by analysing each response against the three criteria, then state your final verdict.

Output one of the following as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: \"My final verdict is: [[A=B]]\"."""

# --- C2 with F2 (label-only): same priority rubric, no reasoning instructions ---
C2_PRIORITY_LABEL_ONLY = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer.

Evaluate the responses on three criteria with explicit priority:
- CORRECTNESS (primary): Are the technical claims, code, calculations, and reasoning accurate? Is the answer factually right?
- HELPFULNESS (constraint): Does the answer address the user's actual request? Is it well-organised and easy to follow?
- SAFETY (constraint): Does the answer avoid harmful, deceptive, or unethical content?

In your final verdict, prioritise correctness. A response that is more helpful but less correct should be rated lower than a response that is correct but less helpful.

Output only one of the following as your final verdict with a label, with no explanation:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: \"[[A=B]]\"."""

# --- D-axis preamble (added to user prompt) ---
D2_SCOPE_PREAMBLE = (
    "The following user prompt is from a coding-assistance task. "
    "Evaluate the responses with that context in mind.\n\n"
)

# --- Arena-Hard v2.0 user prompt template ---
USER_PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{question}\n\n"
    "<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"
)


def get_system_prompt(cell_id: str) -> str:
    """Return system prompt for a cell. cell_id format: 'C{1,2}F{1,2}D{1,2}'."""
    c, f = cell_id[1], cell_id[3]
    if c == "1" and f == "1":
        return OG_ARENA_HARD_PROMPT
    if c == "1" and f == "2":
        return C1_LABEL_ONLY
    if c == "2" and f == "1":
        return C2_PRIORITY_REASONED
    if c == "2" and f == "2":
        return C2_PRIORITY_LABEL_ONLY
    raise ValueError(f"Unknown cell_id: {cell_id}")


def get_user_prompt(cell_id: str, question: str, answer_a: str, answer_b: str) -> str:
    """Build user prompt. cell_id selects whether to add D2 scope preamble."""
    d = cell_id[5]
    body = USER_PROMPT_TEMPLATE.format(question=question, answer_a=answer_a, answer_b=answer_b)
    if d == "1":
        return body
    if d == "2":
        return D2_SCOPE_PREAMBLE + body
    raise ValueError(f"Unknown cell_id: {cell_id}")


L2_CELLS = ["C1F1D1", "C1F1D2", "C1F2D1", "C1F2D2", "C2F1D1", "C2F1D2", "C2F2D1", "C2F2D2"]
ANCHOR_CELL = "C1F1D1"
