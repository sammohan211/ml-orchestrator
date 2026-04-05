"""
Tests for ml_orchestrator/stages/cleaning.py

Coverage targets:
- _apply_renames: rename mapping from data_dictionary state
- _apply_impute: each strategy (mean, median, mode, constant)
- _handle_duplicates: duplicate removal and count
- _handle_columns: routing based on null/warning type
- run(): guard when ingestion not completed
"""
import pytest
import pandas as pd
from unittest.mock import patch


class TestApplyRenames:

    def test_renames_columns_from_data_dictionary(self, sample_df):
        from ml_orchestrator.stages.cleaning import _apply_renames
        state = {"data_dictionary": {"columns": {"gender": {"renamed_to": "sex"}}}}
        result = _apply_renames(sample_df, state)
        assert "sex" in result.columns
        assert "gender" not in result.columns

    def test_no_renames_when_data_dictionary_empty(self, sample_df):
        from ml_orchestrator.stages.cleaning import _apply_renames
        state = {"data_dictionary": {"columns": {}}}
        result = _apply_renames(sample_df, state)
        assert list(result.columns) == list(sample_df.columns)

    def test_ignores_rename_when_renamed_to_is_none(self, sample_df):
        from ml_orchestrator.stages.cleaning import _apply_renames
        state = {"data_dictionary": {"columns": {"gender": {"renamed_to": None}}}}
        result = _apply_renames(sample_df, state)
        assert "gender" in result.columns


class TestApplyImpute:

    def test_impute_mean(self):
        from ml_orchestrator.stages.cleaning import _apply_impute
        df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
        action = {"type": "impute", "column": "x", "strategy": "mean", "value": 7/3}
        result = _apply_impute(df.copy(), "x", action)
        assert result["x"].isnull().sum() == 0

    def test_impute_median(self):
        from ml_orchestrator.stages.cleaning import _apply_impute
        df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
        action = {"type": "impute", "column": "x", "strategy": "median", "value": 2.0}
        result = _apply_impute(df.copy(), "x", action)
        assert result["x"].isnull().sum() == 0

    def test_impute_mode(self):
        from ml_orchestrator.stages.cleaning import _apply_impute
        df = pd.DataFrame({"x": ["a", "b", "a", None]})
        action = {"type": "impute", "column": "x", "strategy": "mode", "value": "a"}
        result = _apply_impute(df.copy(), "x", action)
        assert result["x"].isnull().sum() == 0
        assert result["x"].iloc[3] == "a"

    def test_impute_constant_numeric(self):
        from ml_orchestrator.stages.cleaning import _apply_impute
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        action = {"type": "impute", "column": "x", "strategy": "constant", "value": "0"}
        result = _apply_impute(df.copy(), "x", action)
        assert result["x"].isnull().sum() == 0

    def test_impute_constant_string(self):
        from ml_orchestrator.stages.cleaning import _apply_impute
        df = pd.DataFrame({"x": ["a", None, "c"]})
        action = {"type": "impute", "column": "x", "strategy": "constant", "value": "unknown"}
        result = _apply_impute(df.copy(), "x", action)
        assert result["x"].iloc[1] == "unknown"


class TestHandleDuplicates:

    def test_removes_duplicates(self):
        from ml_orchestrator.stages.cleaning import _handle_duplicates
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        with patch("questionary.confirm") as mock_confirm:
            mock_confirm.return_value.ask.return_value = True
            result_df, actions = _handle_duplicates(df, [])
        assert len(result_df) == 2
        assert any(a["type"] == "remove_duplicates" and a["rows_removed"] == 1 for a in actions)

    def test_skips_when_no_duplicates(self):
        from ml_orchestrator.stages.cleaning import _handle_duplicates
        df = pd.DataFrame({"a": [1, 2, 3]})
        result_df, actions = _handle_duplicates(df, [])
        assert len(result_df) == 3
        assert actions[0]["rows_removed"] == 0


class TestRunGuard:

    def test_returns_state_when_ingestion_not_complete(self, minimal_state):
        from ml_orchestrator.stages.cleaning import run
        minimal_state["stages"]["ingestion"] = "pending"
        result = run(minimal_state)
        assert result is minimal_state
        assert result["stages"]["cleaning"] == "pending"
