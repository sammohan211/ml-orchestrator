"""
Tests for ml_orchestrator/stages/export.py

Coverage targets:
- _export_pipeline: builds a valid sklearn Pipeline and saves it
- _export_pipeline_script: generates valid Python with correct imports
- _export_run_summary: produces a markdown file
- _export_requirements: file contains expected package names
- run(): guard when modeling not completed
"""
import pytest
import joblib
from pathlib import Path
from unittest.mock import patch


class TestExportPipeline:

    def test_saves_pipeline_joblib(self, minimal_state, project_dir):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Plant a preprocessor and model in artifacts
        preprocessor = StandardScaler()
        model = LogisticRegression()
        joblib.dump(preprocessor, project_dir / "artifacts" / "preprocessor.joblib")
        joblib.dump(model, project_dir / "artifacts" / "model.joblib")

        minimal_state["artifacts"]["preprocessor"] = "artifacts/preprocessor.joblib"
        minimal_state["artifacts"]["model"] = "artifacts/model.joblib"

        from ml_orchestrator.stages.export import _export_pipeline
        path = _export_pipeline(minimal_state, project_dir)
        assert path is not None
        assert Path(path).exists()

        loaded = joblib.load(path)
        assert isinstance(loaded, Pipeline)
        assert loaded.steps[0][0] == "preprocessor"
        assert loaded.steps[1][0] == "model"


class TestExportPipelineScript:

    def test_generates_python_file(self, minimal_state, project_dir):
        minimal_state["feature_preparation"] = {
            "target_column": "label",
            "problem_type": "classification",
            "encoding": {"gender": {"strategy": "one_hot", "source": "recommended"}},
            "scaling": {"age": {"strategy": "standard", "source": "recommended"}},
        }
        minimal_state["feature_selection"] = {"selected_features": ["gender_M", "age"]}
        minimal_state["modeling"] = {
            "model_name": "LogisticRegression",
            "model_class": "sklearn.linear_model.LogisticRegression",
            "params": {"C": 1.0},
        }
        minimal_state["tuning"] = {}

        from ml_orchestrator.stages.export import _export_pipeline_script
        path = _export_pipeline_script(minimal_state, project_dir)
        assert path is not None
        content = Path(path).read_text()
        assert "from sklearn.pipeline import Pipeline" in content
        assert "LogisticRegression" in content
        assert "SELECTED_FEATURES" in content

    def test_script_contains_correct_encoder(self, minimal_state, project_dir):
        minimal_state["feature_preparation"] = {
            "target_column": "label",
            "problem_type": "classification",
            "encoding": {"gender": {"strategy": "one_hot", "source": "recommended"}},
            "scaling": {},
        }
        minimal_state["feature_selection"] = {"selected_features": []}
        minimal_state["modeling"] = {
            "model_name": "LogisticRegression",
            "model_class": "sklearn.linear_model.LogisticRegression",
            "params": {},
        }
        minimal_state["tuning"] = {}

        from ml_orchestrator.stages.export import _export_pipeline_script
        path = _export_pipeline_script(minimal_state, project_dir)
        content = Path(path).read_text()
        assert "OneHotEncoder" in content


class TestExportRequirements:

    def test_creates_requirements_file(self, project_dir):
        from ml_orchestrator.stages.export import _export_requirements
        path = _export_requirements(project_dir)
        assert path is not None
        content = Path(path).read_text()
        assert "scikit-learn" in content
        assert "pandas" in content


class TestExportRunSummary:

    def test_creates_markdown_file(self, minimal_state, project_dir):
        from ml_orchestrator.stages.export import _export_run_summary
        path = _export_run_summary(minimal_state, project_dir)
        assert path is not None
        content = Path(path).read_text()
        assert "## Dataset" in content
        assert minimal_state["project"]["name"] in content


class TestRunGuard:

    def test_returns_state_when_modeling_not_complete(self, minimal_state):
        from ml_orchestrator.stages.export import run
        minimal_state["stages"]["modeling"] = "pending"
        result = run(minimal_state)
        assert result is minimal_state
