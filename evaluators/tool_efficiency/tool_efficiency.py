"""Community evaluator: tool_efficiency

Scores tool usage effectiveness. Penalizes duplicate calls (same tool + args),
error responses, and budget overruns.

Config: max_tool_calls (int, default 15), min_tool_calls (int, default 0),
        penalize_duplicates (bool, default true), penalize_errors (bool, default true)
"""

import json
from agentevals_evaluator_sdk import EvalInput, EvalResult, EvalStatus, evaluator


def _call_signature(call) -> str:
    try:
        args_str = json.dumps(call.args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args_str = str(call.args)
    return f"{call.name}::{args_str}"


def _is_error_response(response) -> bool:
    """Check if a tool response indicates an error via its status field."""
    return str(response.status or "").lower() in ("error", "failed", "failure")


@evaluator
def tool_efficiency(input: EvalInput) -> EvalResult:
    max_tool_calls = input.config.get("max_tool_calls", 15)
    min_tool_calls = input.config.get("min_tool_calls", 0)
    penalize_duplicates = input.config.get("penalize_duplicates", True)
    penalize_errors = input.config.get("penalize_errors", True)

    scores: list[float] = []
    details_items: list[str] = []

    for inv in input.invocations:
        tool_calls = inv.intermediate_steps.tool_calls if inv.intermediate_steps else []
        tool_responses = inv.intermediate_steps.tool_responses if inv.intermediate_steps else []
        total = len(tool_calls)

        if total == 0:
            if min_tool_calls > 0:
                scores.append(0.0)
                details_items.append(f"{inv.invocation_id}: no tool calls (min required: {min_tool_calls})")
            else:
                scores.append(1.0)
                details_items.append(f"{inv.invocation_id}: no tool calls (tools optional)")
            continue

        dupes = 0
        if penalize_duplicates:
            seen: dict[str, int] = {}
            for call in tool_calls:
                sig = _call_signature(call)
                seen[sig] = seen.get(sig, 0) + 1
            dupes = sum(c - 1 for c in seen.values() if c > 1)

        errors = sum(1 for r in tool_responses if _is_error_response(r)) if penalize_errors else 0
        useful = max(0, total - dupes - errors)

        efficiency = useful / total
        budget_factor = max(0.0, 1.0 - max(0, total - max_tool_calls) / max_tool_calls) if max_tool_calls > 0 else 0.0
        score = max(0.0, min(1.0, efficiency * budget_factor))
        scores.append(score)

        parts = [f"total={total}", f"useful={useful}"]
        if dupes: parts.append(f"dupes={dupes}")
        if errors: parts.append(f"errors={errors}")
        details_items.append(f"{inv.invocation_id}: {', '.join(parts)}")

    if not scores:
        return EvalResult(
            score=0.0,
            status=EvalStatus.NOT_EVALUATED,
            details={"reason": "no invocations to evaluate"},
        )

    overall = sum(scores) / len(scores)
    return EvalResult(score=overall, per_invocation_scores=scores, details={"issues": details_items})


if __name__ == "__main__":
    tool_efficiency.run()
