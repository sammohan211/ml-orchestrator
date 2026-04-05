"""
Tests for ml_orchestrator/stages/modeling.py

Coverage targets:
- MODEL_REGISTRY: both problem types have entries with required keys
- _parse_param: covered in test_pure_functions.py
- _train: returns fitted model or None on bad params
- _quick_eval: returns correct metric keys per problem type
- run(): guard when feature_preparation not completed
"""
import pytest
import pandas as pd


class TestModelRegistry:

    def test_both_problem_types_present(self):
        from ml_orchestrator.stages.modeling import MODEL_REGISTRY
        assert "classification" in MODEL_REGISTRY
        assert "regression" in MODEL_REGISTRY

    def test_each_entry_has_required_keys(self):
        from ml_orchestrator.stages.modeling import MODEL_REGISTRY
        required = {"name", "sklearn_class", "default_params", "param_grid"}
        for problem_type, models in MODEL_REGISTRY.items():
            for cfg in models:
                missing = required - cfg.keys()
                assert not missing, f"{cfg.get('name')} missing keys: {missing}"

    def test_sklearn_class_is_importable(self):
        import importlib
        from ml_orchestrator.stages.modeling import MODEL_REGISTRY
        for models in MODEL_REGISTRY.values():
            for cfg in models:
                module_path, class_name = cfg["sklearn_class"].rsplit(".", 1)
                module = importlib.import_module(module_path)
                assert hasattr(module, class_name), f"{cfg['sklearn_class']} not importable"


class TestTrain:

    def test_returns_fitted_model(self):
        from ml_orchestrator.stages.modeling import _train, MODEL_REGISTRY
        cfg = MODEL_REGISTRY["classification"][0]  # RandomForestClassifier
        params = {"n_estimators": 10, "random_state": 42}
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        y = pd.Series([i % 2 for i in range(20)])
        model = _train(cfg, params, X, y)
        assert model is not None
        assert hasattr(model, "predict")

    def test_returns_none_on_bad_params(self):
        from ml_orchestrator.stages.modeling import _train, MODEL_REGISTRY
        cfg = MODEL_REGISTRY["classification"][0]
        params = {"n_estimators": "not_a_number"}
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        model = _train(cfg, params, X, y)
        assert model is None


class TestQuickEval:

    def test_classification_returns_accuracy_keys(self):
        from ml_orchestrator.stages.modeling import _quick_eval, MODEL_REGISTRY, _train
        cfg = MODEL_REGISTRY["classification"][0]
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        y = pd.Series([i % 2 for i in range(20)])
        model = _train(cfg, {"n_estimators": 10, "random_state": 42}, X[:16], y[:16])
        metrics = _quick_eval(model, X[:16], X[16:], y[:16], y[16:], "classification")
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics

    def test_regression_returns_r2_keys(self):
        from ml_orchestrator.stages.modeling import _quick_eval, MODEL_REGISTRY, _train
        cfg = MODEL_REGISTRY["regression"][0]
        X = pd.DataFrame({"a": range(20)})
        y = pd.Series([float(i) * 2.5 for i in range(20)])
        model = _train(cfg, {"n_estimators": 10, "random_state": 42}, X[:16], y[:16])
        metrics = _quick_eval(model, X[:16], X[16:], y[:16], y[16:], "regression")
        assert "test_r2" in metrics
        assert "test_mae" in metrics


class TestRunGuard:

    def test_returns_state_when_feature_preparation_not_complete(self, minimal_state):
        from ml_orchestrator.stages.modeling import run
        minimal_state["stages"]["feature_preparation"] = "pending"
        result = run(minimal_state)
        assert result is minimal_state
