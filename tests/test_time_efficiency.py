"""End-to-end tests for the time_efficiency evaluator."""

import json
import subprocess
import sys

import pytest

EVALUATOR = "evaluators/time_efficiency/time_efficiency.py"


def _run(payload: dict) -> dict:
    result = subprocess.run(
        [sys.executable, EVALUATOR],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    return json.loads(result.stdout)


def _make_input(invocations, config=None):
    return {
        "protocol_version": "1.0",
        "metric_name": "time_efficiency",
        "threshold": 0.5,
        "config": config or {},
        "invocations": invocations,
    }


def _inv(inv_id, perf=None):
    return {"invocation_id": inv_id, "performance_metrics": perf}


class TestTimeEfficiencyBasic:
    def test_no_invocations(self):
        result = _run(_make_input([]))
        assert result["status"] == "NOT_EVALUATED"
        assert result["score"] == 0.0

    def test_no_duration_data(self):
        result = _run(_make_input([_inv("inv-1", {})]))
        assert result["status"] == "NOT_EVALUATED"

    def test_no_perf_metrics(self):
        result = _run(_make_input([_inv("inv-1", None)]))
        assert result["status"] == "NOT_EVALUATED"

    def test_zero_duration_perfect_score(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 0}),
        ]))
        assert result["score"] == 1.0

    def test_half_budget(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 60}),
        ]))
        # 1.0 - (60 / 120) = 0.5
        assert result["score"] == pytest.approx(0.5)

    def test_over_budget_clamps_to_zero(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 200}),
        ]))
        assert result["score"] == 0.0


class TestTimeEfficiencyZeroGuard:
    """Regression: division by zero when max_duration_s = 0."""

    def test_zero_max_duration_no_crash(self):
        result = _run(_make_input(
            [_inv("inv-1", {"duration_s": 10})],
            config={"max_duration_s": 0},
        ))
        assert result["score"] == 0.0

    def test_negative_max_duration_no_crash(self):
        result = _run(_make_input(
            [_inv("inv-1", {"duration_s": 10})],
            config={"max_duration_s": -5},
        ))
        assert result["score"] == 0.0

    def test_zero_duration_with_zero_max(self):
        result = _run(_make_input(
            [_inv("inv-1", {"duration_s": 0})],
            config={"max_duration_s": 0},
        ))
        assert result["score"] == 0.0


class TestTimeEfficiencyZeroValues:
    """Regression: `or` operator previously dropped zero duration values."""

    def test_zero_duration_s_not_dropped(self):
        """duration_s=0 should be used, not fall back to duration key."""
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 0, "duration": 999}),
        ]))
        # duration_s=0 gives score 1.0; if it fell back to 999, score would be 0.0
        assert result["score"] == 1.0


class TestTimeEfficiencyAliases:
    def test_duration_alias(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration": 60}),
        ]))
        assert result["score"] == pytest.approx(0.5)

    def test_duration_s_takes_precedence(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 0, "duration": 120}),
        ]))
        assert result["score"] == 1.0


class TestTimeEfficiencyConfig:
    def test_custom_max_duration(self):
        result = _run(_make_input(
            [_inv("inv-1", {"duration_s": 5})],
            config={"max_duration_s": 10},
        ))
        assert result["score"] == pytest.approx(0.5)


class TestTimeEfficiencyMultipleInvocations:
    def test_average_across_invocations(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 0}),    # score = 1.0
            _inv("inv-2", {"duration_s": 120}),   # score = 0.0
        ]))
        assert result["score"] == pytest.approx(0.5)
        assert len(result["per_invocation_scores"]) == 2

    def test_mixed_with_missing(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 0}),  # score = 1.0
            _inv("inv-2", None),                # score = 0.0 (missing)
        ]))
        assert result["status"] is None
        assert result["score"] == pytest.approx(0.5)

    def test_uses_issues_key_in_details(self):
        result = _run(_make_input([
            _inv("inv-1", {"duration_s": 10}),
        ]))
        assert "issues" in result["details"]
