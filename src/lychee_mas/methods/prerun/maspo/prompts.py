"""Course-authored prompt texts for the MASPO algorithm interfaces.

The optimization algorithm follows MASPO, arXiv:2605.06623. These concise
prompts were authored for this integration and replace upstream verbatim assets.
"""
ROLE_DESCRIPTIONS = {
    "solver": "Solve the task and state a justified answer.",
    "analyst": "Derive relationships and check the meaning of the task.",
    "programmer": "Express calculations as small verifiable arithmetic programs.",
    "inspector": "Identify and repair mistakes in available reasoning.",
    "final": "Resolve the evidence and provide the final task answer.",
}
COMPRESS_PROMPT = "Summarize the following reasoning while preserving its key claims and answer:\n{raw}"
OPTIMIZATION_REQUIREMENT = (
    "Improve the correctness and usefulness of the team's final answer. "
    "Keep instructions concise and task-general; do not encode answers to training examples. "
    "Preserve the role and required output format."
)
PROMPT_OPTIMIZE_TEMPLATE = """Improve one agent instruction in a cooperating team.
Role: {agent_type}. {role_description}
Requirements: {requirements}
Current instruction:
{prompt}
Observed training examples and agent outputs:
{samples}
Diagnose one concrete weakness and write a reusable improvement. Preserve literal
placeholders {{question}} and {{context}}. Do not copy example-specific answers.
Return only <prompt>the complete replacement instruction</prompt>.
"""
ANSWER_EVALUATE_TEMPLATE = """Compare two outputs for the assigned role.
Role: {agent_type}. {role_description}
Task: {question}
Criteria: {requirement}
Output A:
{Answer_A}
Output B:
{Answer_B}
Check correctness independently. Prefer the more useful and correct output.
If tied choose B. End with exactly <choose>A</choose> or <choose>B</choose>.
"""
INTERMEDIATE_COMPARE_TEMPLATE = """Compare intermediate messages for a cooperating team.
Task: {question}
Message A:
{output_a}
Message B:
{output_b}
Check factual and mathematical validity, useful information, and whether a downstream
agent can use it to solve the task. Do not reward length by itself. If tied choose B.
Give one short reason and end with <choose>A</choose> or <choose>B</choose>.
"""
FINAL_ANSWER_COMPARE_TEMPLATE = """Compare two final task solutions independently.
Task: {question}
Criteria: {requirement}
Solution A:
{Answer_A}
Solution B:
{Answer_B}
Prefer the correct, well-supported answer; if equally good choose B.
Give one short reason and end with <choose>A</choose> or <choose>B</choose>.
"""
def role_description(role):
    return ROLE_DESCRIPTIONS[role]

def seed_template(role):
    return role_description(role) + "\nTask: {question}\nPrior messages: {context}"
