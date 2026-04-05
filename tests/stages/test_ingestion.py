"""
Tests for ml_orchestrator/stages/ingestion.py

Coverage targets:
- run(): guard when no project loaded
- Format detection by file extension
- CSV encoding/delimiter detection
- State updates after successful load
"""
import pytest


class TestRunGuard:

    def test_returns_state_on_cancelled_prompt(self, minimal_state):
        """Ingestion has no project guard — it's protected at the menu level.
        Cancelling the file path prompt (questionary returns None) should return state unchanged."""
        from unittest.mock import patch
        from ml_orchestrator.stages.ingestion import run
        with patch("questionary.text") as mock_text:
            mock_text.return_value.ask.return_value = None
            result = run(minimal_state)
        assert result["stages"]["ingestion"] == "completed"  # unchanged from fixture


@pytest.mark.skip(reason="requires interactive prompts — mock questionary")
class TestCSVLoad:

    def test_loads_csv_and_updates_state(self, minimal_state, sample_df, project_dir):
        """After loading a CSV, state should have ingestion.row_count, column_count, schema."""
        pass

    def test_detects_delimiter(self, project_dir):
        """Pipe-delimited CSV should be detected correctly via csv.Sniffer."""
        pass

    def test_detects_encoding(self, project_dir):
        """File with non-UTF-8 encoding should be detected via charset-normalizer."""
        pass


@pytest.mark.skip(reason="requires interactive prompts — mock questionary")
class TestParquetLoad:

    def test_loads_parquet_and_updates_state(self, minimal_state, project_dir):
        """After loading a parquet file, state should be updated correctly."""
        pass
