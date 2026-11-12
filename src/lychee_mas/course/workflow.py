"""Five persistent roles, two communication rounds, and message-level ADv2 gates."""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable

from ..core.types import AgentSpec
from ..methods.prerun.maspo.executor import CachedExecutor, InferenceCache, format_agent_prompt
from ..methods.prerun.maspo.textops import extract_answer
from ..methods.runtime.agentdropout_v2 import AgentDropoutV2Controller, Indicator
from ..plugins.prerun.graphview import GraphView
from .io import PURPOSE

AGENTS = ("A", "B", "C", "D", "E")
ROLE_NAMES = ("solver", "analyst", "programmer", "inspector", "solver")


def safe_arithmetic(code: str) -> str:
    """Interpret a small arithmetic language; never execute arbitrary generated Python."""
    tree = ast.parse(code)
    if len(list(ast.walk(tree))) > 300:
        raise ValueError("arithmetic program is too long")
    env: dict = {}
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                  ast.Mod: operator.mod, ast.Pow: operator.pow}
    calls = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum,
             "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor}

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            out = node.value
        elif isinstance(node, ast.Name) and node.id in env:
            out = env[node.id]
        elif isinstance(node, (ast.List, ast.Tuple)):
            out = [value(item) for item in node.elts]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            out = (-1 if isinstance(node.op, ast.USub) else 1) * value(node.operand)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = value(node.left), value(node.right)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("binary operations require numbers")
            if isinstance(node.op, ast.Pow) and abs(right) > 20:
                raise ValueError("exponent exceeds arithmetic limit")
            out = operations[type(node.op)](left, right)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in calls:
            if node.keywords:
                raise ValueError("keyword arguments are unsupported")
            out = calls[node.func.id](*[value(arg) for arg in node.args])
        else:
            raise ValueError(f"unsupported arithmetic syntax: {type(node).__name__}")
        if isinstance(out, (int, float)) and (not math.isfinite(out) or abs(out) > 1e30):
            raise ValueError("arithmetic value exceeds numeric limits")
        return out

    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("use arithmetic assignments only")
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id.startswith("_"):
            raise ValueError("assignment target must be a plain variable")
        env[target.id] = value(statement.value)
    if "answer" not in env or not isinstance(env["answer"], (int, float)):
        raise ValueError("program must assign a numeric answer")
    return str(env["answer"])


def tool_result(raw: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.S)
    if not blocks:
        return raw + "\n[Arithmetic tool: no Python block supplied.]"
    try:
        answer = safe_arithmetic(blocks[0])
        return raw + f"\n[Arithmetic tool result: {answer}]"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as error:
        return raw + f"\n[Arithmetic tool error: {error}]"


def seed_prompts(benchmark: str) -> dict[str, str]:
    choice = benchmark in ("arc_easy", "openbookqa")
    finish = ("Finish with <answer>the single option letter</answer>." if choice else
              "Finish with <answer>the exact numeric or mathematical answer</answer>.")
    instructions = {
        "A": "Solve the task independently, using prior messages as fallible suggestions.",
        "B": "Analyze the variables, constraints and relevant knowledge before solving the task.",
        "C": ("Compare all options using the relevant scientific facts and eliminate distractors."
              if choice else
              "Verify the arithmetic with a short Python block containing assignments only. "
              "Use numeric literals, variables, + - * / // % **, abs round min max sum sqrt ceil floor. "
              "Do not use imports, functions, attributes or loops. Assign the result to answer. "
              "After the block explain whether the calculation supports the solution."),
        "D": "Check upstream claims against the original task. Correct the most important error, if any.",
        "E": "Solve the task again, reconcile conflicting suggestions, and justify your chosen answer.",
    }
    return {
        name: (f"You are agent {name}, the {ROLE_NAMES[i]}. {instructions[name]} "
               "Keep the reasoning concise (at most 180 words). " + finish +
               "\nTask: {question}\nAvailable prior messages:\n{context}")
        for i, name in enumerate(AGENTS)
    }


def make_view(benchmark: str, rounds: int = 2) -> GraphView:
    if rounds != 2:
        raise ValueError("the course protocol uses exactly two rounds")
    prompts = seed_prompts(benchmark)
    names, specs, predecessors = [], {}, {}
    for turn in range(1, rounds + 1):
        for index, agent in enumerate(AGENTS):
            name = agent if turn == 1 else f"{agent}_r{turn}"
            spatial = [a if turn == 1 else f"{a}_r{turn}" for a in AGENTS[:index]]
            temporal = list(AGENTS[:index + 1]) if turn == 2 else []
            names.append(name)
            predecessors[name] = spatial + temporal
            specs[name] = AgentSpec(name=name, role=ROLE_NAMES[index],
                                   system_prompt=prompts[agent],
                                   meta={"predecessors": predecessors[name], "prompt_source": agent,
                                         "round": turn, "agent": agent, "benchmark": benchmark})
    names.append("final")
    predecessors["final"] = [f"{a}_r2" for a in AGENTS]
    output_rule = ("the option letter" if benchmark in ("arc_easy", "openbookqa")
                   else "the exact numeric or mathematical answer")
    specs["final"] = AgentSpec(
        name="final", role="final",
        system_prompt=("Resolve the available team evidence against the original task. "
                       "Give a brief verification and finish with <answer>" + output_rule +
                       "</answer>.\nTask: {question}\nAccepted final-round messages:\n{context}"),
        meta={"predecessors": predecessors["final"], "benchmark": benchmark})
    return GraphView(names, specs, predecessors, "final")


def fixed_indicators(benchmark: str) -> list[Indicator]:
    choice = benchmark in ("arc_easy", "openbookqa")
    return [
        Indicator("task_fidelity", "Use the original task facts, units, conditions and requested quantity; do not introduce contradictory assumptions."),
        Indicator("reasoning_validity", ("Scientific facts and logical implications must support the chosen option; an upstream assertion alone is not evidence."
                                         if choice else "Arithmetic, algebra and logical steps must be valid and support the stated result; check any reported tool result.")),
        Indicator("answer_consistency", "A stated answer must agree with the message's own derivation and available evidence; do not propagate a known unresolved contradiction."),
    ]


class CourseExecutor(CachedExecutor):
    """MASPO cached execution and live execution share exactly the same message construction."""

    def __init__(self, view, agent_llm: Callable, prompt_map=None, gate=None):
        super().__init__(view, agent_llm, prompt_map)
        self.gate: AgentDropoutV2Controller | None = gate
        self.records: list[dict] = []

    def _template(self, name: str) -> str:
        source = self.view.specs[name].meta.get("prompt_source", name)
        return self.prompt_map.get(source, self.view.specs[source].system_prompt)

    async def _run_node(self, name: str, question: str, cache: InferenceCache) -> None:
        spec = self.view.specs[name]
        visible = [p for p in self.view.predecessors[name]
                   if cache.node_outputs_raw.get(p)]
        context = "\n\n".join(f"[{p}]\n{cache.node_outputs_raw[p]}" for p in visible)
        prompt = format_agent_prompt(self._template(name), question, context)
        use_tool = spec.meta.get("agent") == "C" and spec.meta["benchmark"] in ("gsm8k", "math")

        async def generate(text: str) -> str:
            token = PURPOSE.set(name)
            try:
                output = await self.agent_llm(text)
                return tool_result(output) if use_tool else output
            finally:
                PURPOSE.reset(token)

        raw = await generate(prompt)
        if self.gate:
            async def regenerate(feedback: str) -> str:
                return await generate(prompt + "\n\nVerifier feedback:\n" + feedback)
            result = await self.gate.process(raw, task=question, agent=name,
                                             regenerate=regenerate, context=context)
            accepted = result.content or ""
            gate_data = result.to_dict()
        else:
            accepted, gate_data = raw, None
        # Rejected drafts are absent from all spatial, temporal and final input contexts.
        cache.set_node_data(name, context, accepted, accepted)
        self.records.append({"node": name, "agent": spec.meta.get("agent", "final"),
                             "round": spec.meta.get("round"), "visible_sources": visible,
                             "input_context": context, "initial_output": raw,
                             "accepted_output": accepted or None, "gate": gate_data})

    async def run(self, question: str):
        # Execution order and communication edges are separate. In particular,
        # A_r2 sees only A, but still waits until the whole first round completes.
        cache = InferenceCache()
        cache.question = question
        for name in self.view.names:
            await self._run_node(name, question, cache)
        return extract_answer(cache.node_outputs_raw.get(self.view.terminal, "")), cache


def langgraph_from_view(view: GraphView, executor: CourseExecutor):
    """Optional native StateGraph carrier using LycheeMAS AgentSpec metadata."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(dict)
    for name in view.names:
        async def node(state, name=name):
            cache = state["cache"]
            await executor._run_node(name, state["question"], cache)
            return {"question": state["question"], "cache": cache}
        graph.add_node(name, node, metadata={"agent_spec": view.specs[name]})
    graph.add_edge(START, view.names[0])
    for before, after in zip(view.names, view.names[1:]):
        graph.add_edge(before, after)
    graph.add_edge(view.names[-1], END)
    return graph


async def run_langgraph(view: GraphView, executor: CourseExecutor, question: str):
    cache = InferenceCache()
    cache.question = question
    state = await langgraph_from_view(view, executor).compile().ainvoke(
        {"question": question, "cache": cache})
    cache = state["cache"]
    return extract_answer(cache.node_outputs_raw.get("final", "")), cache
