"""
Unit tests for pure functions that require no I/O, mocking, or user prompts.
These should always run fast and never touch the filesystem.
"""
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# modeling._parse_param
# ---------------------------------------------------------------------------

class TestParseParam:
    from ml_orchestrator.stages.modeling import _parse_param

    def test_none_string(self):
        from ml_orchestrator.stages.modeling import _parse_param
        assert _parse_param("None") is None

    def test_true_string(self):
        from ml_orchestrator.stages.modeling import _parse_param
        assert _parse_param("True") is True

    def test_false_string(self):
        from ml_orchestrator.stages.modeling import _parse_param
        assert _parse_param("False") is False

    def test_integer(self):
        from ml_orchestrator.stages.modeling import _parse_param
        assert _parse_param("100") == 100
        assert isinstance(_parse_param("100"), int)

    def test_float(self):
        from ml_orchestrator.stages.modeling import _parse_param
        assert _parse_param("0.1") == pytest.approx(0.1)
        assert isinstance(_parse_param("0.1"), float)

    def test_string_fallback(self):
        from ml_orchestrator.stages.modeling import _parse_param
        assert _parse_param("lbfgs") == "lbfgs"


# ---------------------------------------------------------------------------
# feature_preparation._infer_problem_type
# ---------------------------------------------------------------------------

class TestInferProblemType:

    def test_string_target_is_classification(self):
        from ml_orchestrator.stages.feature_preparation import _infer_problem_type
        s = pd.Series(["yes", "no", "yes", "no"])
        assert _infer_problem_type(s) == "classification"

    def test_binary_int_target_is_classification(self):
        from ml_orchestrator.stages.feature_preparation import _infer_problem_type
        s = pd.Series([0, 1, 0, 1, 1])
        assert _infer_problem_type(s) == "classification"

    def test_low_cardinality_numeric_is_classification(self):
        from ml_orchestrator.stages.feature_preparation import _infer_problem_type
        s = pd.Series(list(range(10)) * 5)  # 10 unique values
        assert _infer_problem_type(s) == "classification"

    def test_high_cardinality_numeric_is_regression(self):
        from ml_orchestrator.stages.feature_preparation import _infer_problem_type
        s = pd.Series([float(i) * 1.5 for i in range(100)])  # 100 unique values
        assert _infer_problem_type(s) == "regression"


# ---------------------------------------------------------------------------
# feature_preparation._clean_feature_names
# ---------------------------------------------------------------------------

class TestCleanFeatureNames:

    def test_strips_one_hot_prefix(self):
        from ml_orchestrator.stages.feature_preparation import _clean_feature_names
        raw = ["one_hot__gender_M", "one_hot__gender_F"]
        assert _clean_feature_names(raw) == ["gender_M", "gender_F"]

    def test_strips_standard_prefix(self):
        from ml_orchestrator.stages.feature_preparation import _clean_feature_names
        raw = ["standard__age", "standard__income"]
        assert _clean_feature_names(raw) == ["age", "income"]

    def test_strips_remainder_prefix(self):
        from ml_orchestrator.stages.feature_preparation import _clean_feature_names
        raw = ["remainder__subscribed"]
        assert _clean_feature_names(raw) == ["subscribed"]

    def test_no_prefix_unchanged(self):
        from ml_orchestrator.stages.feature_preparation import _clean_feature_names
        raw = ["age", "income"]
        assert _clean_feature_names(raw) == ["age", "income"]

    def test_mixed(self):
        from ml_orchestrator.stages.feature_preparation import _clean_feature_names
        raw = ["one_hot__gender_M", "standard__age", "remainder__subscribed"]
        assert _clean_feature_names(raw) == ["gender_M", "age", "subscribed"]


# ---------------------------------------------------------------------------
# feature_preparation._valid_float
# ---------------------------------------------------------------------------

class TestValidFloat:

    def test_valid_value(self):
        from ml_orchestrator.stages.feature_preparation import _valid_float
        assert _valid_float("0.2", 0.0, 0.5) is True

    def test_boundary_values(self):
        from ml_orchestrator.stages.feature_preparation import _valid_float
        assert _valid_float("0.0", 0.0, 0.5) is True
        assert _valid_float("0.5", 0.0, 0.5) is True

    def test_out_of_range(self):
        from ml_orchestrator.stages.feature_preparation import _valid_float
        result = _valid_float("0.9", 0.0, 0.5)
        assert result is not True

    def test_non_numeric(self):
        from ml_orchestrator.stages.feature_preparation import _valid_float
        result = _valid_float("abc", 0.0, 0.5)
        assert result is not True


# ---------------------------------------------------------------------------
# tuning._count_combinations
# ---------------------------------------------------------------------------

class TestCountCombinations:

    def test_single_param(self):
        from ml_orchestrator.stages.tuning import _count_combinations
        assert _count_combinations({"n_estimators": [100, 200, 300]}) == 3

    def test_multiple_params(self):
        from ml_orchestrator.stages.tuning import _count_combinations
        grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
        assert _count_combinations(grid) == 6

    def test_empty_grid(self):
        from ml_orchestrator.stages.tuning import _count_combinations
        assert _count_combinations({}) == 1


# ---------------------------------------------------------------------------
# settings._cast and _validate
# ---------------------------------------------------------------------------

class TestSettingsCast:

    def test_cast_int(self):
        from ml_orchestrator.stages.settings import _cast
        assert _cast("42", int) == 42
        assert isinstance(_cast("42", int), int)

    def test_cast_float(self):
        from ml_orchestrator.stages.settings import _cast
        assert _cast("0.2", float) == pytest.approx(0.2)

    def test_cast_bool_true_variants(self):
        from ml_orchestrator.stages.settings import _cast
        for v in ("true", "yes", "1"):
            assert _cast(v, bool) is True

    def test_cast_bool_false_variants(self):
        from ml_orchestrator.stages.settings import _cast
        for v in ("false", "no", "0"):
            assert _cast(v, bool) is False


class TestSettingsValidate:

    def test_valid_int(self):
        from ml_orchestrator.stages.settings import _validate
        assert _validate("42", int) is True

    def test_invalid_int(self):
        from ml_orchestrator.stages.settings import _validate
        assert _validate("abc", int) is not True

    def test_valid_bool(self):
        from ml_orchestrator.stages.settings import _validate
        assert _validate("true", bool) is True
        assert _validate("false", bool) is True

    def test_invalid_bool(self):
        from ml_orchestrator.stages.settings import _validate
        assert _validate("maybe", bool) is not True
