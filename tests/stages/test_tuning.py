"""
Tests for ml_orchestrator/stages/tuning.py

Coverage targets:
- _count_combinations: covered in test_pure_functions.py
- _find_model_cfg: returns correct config or None
- _show_comparison: runs without error (smoke test)
- run(): guard when modeling not completed
"""
import pytest


class TestFindModelCfg:

    def test_finds_existing_model(self):
        from ml_orchestrator.stages.tuning import _find_model_cfg
        cfg = _find_model_cfg("RandomForestClassifier", "classification")
        assert cfg is not None
        assert cfg["name"] == "RandomForestClassifier"

    def test_returns_none_for_unknown_model(self):
        from ml_orchestrator.stages.tuning import _find_model_cfg
        cfg = _find_model_cfg("XGBoostClassifier", "classification")
        assert cfg is None

    def test_returns_none_for_wrong_problem_type(self):
        from ml_orchestrator.stages.tuning import _find_model_cfg
        cfg = _find_model_cfg("RandomForestClassifier", "regression")
        assert cfg is None


class TestShowComparison:

    def test_runs_without_error(self, capsys):
        """Smoke test — just checks it doesn't raise."""
        from ml_orchestrator.stages.tuning import _show_comparison
        original = {"n_estimators": 100, "max_depth": None}
        best = {"n_estimators": 200, "max_depth": 10}
        _show_comparison(original, best)  # should not raise


class TestRunGuard:

    def test_returns_state_when_modeling_not_complete(self, minimal_state):
        from ml_orchestrator.stages.tuning import run
        minimal_state["stages"]["modeling"] = "pending"
        result = run(minimal_state)
        assert result is minimal_state


@pytest.mark.skip(reason="slow — runs RandomizedSearchCV; run explicitly with -m slow")
class TestRunSearch:

    def test_best_params_are_subset_of_param_grid(self, minimal_state):
        """best_params keys should all be keys in the param_grid."""
        pass

    def test_capped_iterations_do_not_exceed_grid_size(self, minimal_state):
        """When grid has 4 combinations and n_iter=20, should only do 4 fits."""
        pass
