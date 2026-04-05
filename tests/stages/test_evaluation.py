"""
Tests for ml_orchestrator/stages/evaluation.py

Coverage targets:
- _evaluate_classification: returns correct metric keys, handles binary and multiclass
- _evaluate_regression: returns correct metric keys, handles zero-division in MAPE
- _show_confusion_matrix: smoke test (no exception)
- run(): guard when modeling not completed
"""
import pytest
import pandas as pd
import numpy as np


class TestEvaluateClassification:

    def _make_model(self):
        from sklearn.ensemble import RandomForestClassifier
        X = pd.DataFrame({"a": range(40), "b": range(40)})
        y = pd.Series([i % 2 for i in range(40)])
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        return model, X[30:], y[30:]

    def test_returns_required_keys(self):
        from ml_orchestrator.stages.evaluation import _evaluate_classification
        model, X_test, y_test = self._make_model()
        metrics, report = _evaluate_classification(model, X_test, y_test)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics

    def test_includes_roc_auc_for_binary(self):
        from ml_orchestrator.stages.evaluation import _evaluate_classification
        model, X_test, y_test = self._make_model()
        metrics, _ = _evaluate_classification(model, X_test, y_test)
        assert "roc_auc" in metrics

    def test_metrics_in_valid_range(self):
        from ml_orchestrator.stages.evaluation import _evaluate_classification
        model, X_test, y_test = self._make_model()
        metrics, _ = _evaluate_classification(model, X_test, y_test)
        for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            assert 0.0 <= metrics[key] <= 1.0


class TestEvaluateRegression:

    def _make_regression_model(self):
        from sklearn.ensemble import RandomForestRegressor
        X = pd.DataFrame({"a": range(40)})
        y = pd.Series([float(i) * 2.5 for i in range(40)])
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X[:30], y[:30])
        return model, X[30:], y[30:]

    def test_returns_required_keys(self):
        from ml_orchestrator.stages.evaluation import _evaluate_regression
        model, X_test, y_test = self._make_regression_model()
        metrics, _ = _evaluate_regression(model, X_test, y_test)
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_rmse_equals_sqrt_mse(self):
        import math
        from ml_orchestrator.stages.evaluation import _evaluate_regression
        model, X_test, y_test = self._make_regression_model()
        metrics, _ = _evaluate_regression(model, X_test, y_test)
        assert metrics["rmse"] == pytest.approx(math.sqrt(metrics["mse"]), rel=1e-4)

    def test_handles_zero_targets_in_mape(self):
        """MAPE should not raise when target contains zeros."""
        from ml_orchestrator.stages.evaluation import _evaluate_regression
        from sklearn.linear_model import LinearRegression
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series([0.0] * 10)
        model = LinearRegression().fit(X, y)
        metrics, _ = _evaluate_regression(model, X, y)
        assert metrics is not None


class TestRunGuard:

    def test_returns_state_when_modeling_not_complete(self, minimal_state):
        from ml_orchestrator.stages.evaluation import run
        minimal_state["stages"]["modeling"] = "pending"
        result = run(minimal_state)
        assert result is minimal_state
