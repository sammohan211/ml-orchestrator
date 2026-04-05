"""
Tests for ml_orchestrator/stages/feature_selection.py

Coverage targets:
- _valid_threshold: boundary and invalid input validation
- _model_based_scores: returns dict of feature->score, all values in [0, 1]
- _correlation_scores: returns dict, non-numeric columns score 0.0
- run(): guard when feature_preparation not completed
"""
import pytest
import pandas as pd


class TestValidThreshold:

    def test_valid_values(self):
        from ml_orchestrator.stages.feature_selection import _valid_threshold
        assert _valid_threshold("0.0") is True
        assert _valid_threshold("0.5") is True
        assert _valid_threshold("1.0") is True

    def test_out_of_range(self):
        from ml_orchestrator.stages.feature_selection import _valid_threshold
        assert _valid_threshold("-0.1") is not True
        assert _valid_threshold("1.1") is not True

    def test_non_numeric(self):
        from ml_orchestrator.stages.feature_selection import _valid_threshold
        assert _valid_threshold("abc") is not True


class TestModelBasedScores:

    def test_returns_score_for_each_feature(self, minimal_state):
        from ml_orchestrator.stages.feature_selection import _model_based_scores
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([0, 1, 0, 1, 0])
        scores, params = _model_based_scores(X, y, "classification", minimal_state["settings"])
        assert set(scores.keys()) == {"a", "b"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        assert abs(sum(scores.values()) - 1.0) < 1e-6  # importances sum to 1

    def test_scores_sorted_descending(self, minimal_state):
        from ml_orchestrator.stages.feature_selection import _model_based_scores
        X = pd.DataFrame({"a": range(20), "b": range(20), "c": [0] * 20})
        y = pd.Series([i % 2 for i in range(20)])
        scores, _ = _model_based_scores(X, y, "classification", minimal_state["settings"])
        values = list(scores.values())
        assert values == sorted(values, reverse=True)


class TestCorrelationScores:

    def test_non_numeric_columns_score_zero(self):
        from ml_orchestrator.stages.feature_selection import _correlation_scores
        X = pd.DataFrame({"age": [25, 30, 35], "gender": ["M", "F", "M"]})
        y = pd.Series([1, 0, 1])
        scores, _ = _correlation_scores(X, y)
        assert scores["gender"] == 0.0

    def test_numeric_columns_have_nonzero_score(self):
        from ml_orchestrator.stages.feature_selection import _correlation_scores
        X = pd.DataFrame({"age": [25.0, 30.0, 35.0, 40.0, 45.0]})
        y = pd.Series([0, 0, 1, 1, 1])
        scores, _ = _correlation_scores(X, y)
        assert scores["age"] > 0.0


class TestRunGuard:

    def test_returns_state_when_feature_preparation_not_complete(self, minimal_state):
        from ml_orchestrator.stages.feature_selection import run
        minimal_state["stages"]["feature_preparation"] = "pending"
        result = run(minimal_state)
        assert result is minimal_state
