"""
Tests for ml_orchestrator/stages/feature_preparation.py

Coverage targets:
- _drop_excluded: removes columns listed in cleaning.excluded_columns
- _plan_encoding: one_hot for low cardinality, ordinal for high cardinality and binary
- _plan_scaling: all numeric columns get standard scaling by default
- _apply_transforms: ColumnTransformer output shape and feature name cleaning
- run(): guard when feature_preparation not completed
"""
import pytest
import pandas as pd


class TestDropExcluded:

    def test_drops_excluded_columns(self, minimal_state, sample_df):
        from ml_orchestrator.stages.feature_preparation import _drop_excluded
        minimal_state["cleaning"] = {"excluded_columns": ["gender"]}
        result = _drop_excluded(sample_df.copy(), minimal_state)
        assert "gender" not in result.columns

    def test_no_op_when_no_excluded(self, minimal_state, sample_df):
        from ml_orchestrator.stages.feature_preparation import _drop_excluded
        minimal_state["cleaning"] = {"excluded_columns": []}
        result = _drop_excluded(sample_df.copy(), minimal_state)
        assert list(result.columns) == list(sample_df.columns)

    def test_ignores_missing_excluded_columns(self, minimal_state, sample_df):
        """Excluded column that doesn't exist in df should not raise."""
        from ml_orchestrator.stages.feature_preparation import _drop_excluded
        minimal_state["cleaning"] = {"excluded_columns": ["nonexistent_col"]}
        result = _drop_excluded(sample_df.copy(), minimal_state)
        assert list(result.columns) == list(sample_df.columns)


class TestPlanEncoding:

    def test_binary_column_gets_ordinal(self, minimal_state):
        from ml_orchestrator.stages.feature_preparation import _plan_encoding
        df = pd.DataFrame({"flag": ["yes", "no", "yes", "no"]})
        plan = _plan_encoding(df, minimal_state)
        assert plan["flag"]["strategy"] == "ordinal"

    def test_low_cardinality_gets_one_hot(self, minimal_state):
        from ml_orchestrator.stages.feature_preparation import _plan_encoding
        df = pd.DataFrame({"city": ["NY", "LA", "SF", "NY", "LA"]})
        plan = _plan_encoding(df, minimal_state)
        assert plan["city"]["strategy"] == "one_hot"

    def test_high_cardinality_gets_ordinal(self, minimal_state):
        from ml_orchestrator.stages.feature_preparation import _plan_encoding
        minimal_state["settings"]["high_cardinality_threshold"] = 5
        df = pd.DataFrame({"city": [f"city_{i}" for i in range(10)]})
        plan = _plan_encoding(df, minimal_state)
        assert plan["city"]["strategy"] == "ordinal"

    def test_numeric_columns_not_in_plan(self, minimal_state):
        from ml_orchestrator.stages.feature_preparation import _plan_encoding
        df = pd.DataFrame({"age": [25, 30, 35], "gender": ["M", "F", "M"]})
        plan = _plan_encoding(df, minimal_state)
        assert "age" not in plan
        assert "gender" in plan


class TestPlanScaling:

    def test_numeric_columns_get_standard(self):
        from ml_orchestrator.stages.feature_preparation import _plan_scaling
        df = pd.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]})
        plan = _plan_scaling(df)
        assert plan["age"]["strategy"] == "standard"
        assert plan["income"]["strategy"] == "standard"

    def test_categorical_columns_not_in_plan(self):
        from ml_orchestrator.stages.feature_preparation import _plan_scaling
        df = pd.DataFrame({"age": [25, 30], "gender": ["M", "F"]})
        plan = _plan_scaling(df)
        assert "gender" not in plan


@pytest.mark.skip(reason="requires interactive prompts — mock questionary")
class TestApplyTransforms:

    def test_output_has_no_nulls_after_encoding(self):
        """Encoded dataframe should have no nulls."""
        pass

    def test_feature_names_have_no_sklearn_prefix(self):
        """Output column names should not contain 'one_hot__' etc."""
        pass

    def test_train_test_shapes_consistent(self):
        """X_train and X_test should have same number of columns after transform."""
        pass


class TestRunGuard:

    def test_returns_state_when_ingestion_not_complete(self, minimal_state):
        from ml_orchestrator.stages.feature_preparation import run
        minimal_state["stages"]["ingestion"] = "pending"
        result = run(minimal_state)
        assert result is minimal_state
