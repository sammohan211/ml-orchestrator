import joblib
from datetime import datetime
from pathlib import Path

import questionary
from rich import box
from rich.console import Console
from rich.table import Table

from ml_orchestrator.state.manager import save_state, update_state

console = Console()


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Export Artifacts ──[/bold blue]\n")

    if state.get("stages", {}).get("modeling") != "completed":
        console.print("[yellow]No model found. Please run 'Train Baseline Model' first (option 8).[/yellow]")
        return state

    project_dir = Path(state["project"]["project_dir"])

    exports = _select_exports()
    if exports is None:
        return state

    exported: list[str] = []

    if "pipeline" in exports:
        path = _export_pipeline(state, project_dir)
        if path:
            exported.append(path)

    if "script" in exports:
        path = _export_pipeline_script(state, project_dir)
        if path:
            exported.append(path)

    if "summary" in exports:
        path = _export_run_summary(state, project_dir)
        if path:
            exported.append(path)

    if "requirements" in exports:
        path = _export_requirements(project_dir)
        if path:
            exported.append(path)

    if not exported:
        console.print("[yellow]Nothing exported.[/yellow]")
        return state

    console.print("\n[bold]Exported Files[/bold]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("File")
    table.add_column("Location")
    for p in exported:
        rel = Path(p).relative_to(project_dir)
        table.add_row(Path(p).name, str(rel))
    console.print(table)

    artifact_map = {Path(p).stem: str(Path(p).relative_to(project_dir)) for p in exported}
    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"].update(artifact_map)

    summary = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "exported_files": [str(Path(p).relative_to(project_dir)) for p in exported],
    }
    state = update_state(state, export=summary)
    state["stages"]["export"] = "completed"
    save_state(state)

    console.print("[green]Export complete.[/green]\n")
    return state


# ---------------------------------------------------------------------------
# Export selection
# ---------------------------------------------------------------------------

def _select_exports() -> list[str] | None:
    choices = questionary.checkbox(
        "Select artifacts to export:",
        choices=[
            questionary.Choice("Fitted sklearn Pipeline (pipeline.joblib)", value="pipeline", checked=True),
            questionary.Choice("Pipeline script with comments (pipeline_script.py)", value="script", checked=True),
            questionary.Choice("Run summary report (run_summary.md)", value="summary", checked=True),
            questionary.Choice("Requirements file (requirements.txt)", value="requirements", checked=True),
        ],
    ).ask()
    return choices


# ---------------------------------------------------------------------------
# Pipeline joblib
# ---------------------------------------------------------------------------

def _export_pipeline(state: dict, project_dir: Path) -> str | None:
    from sklearn.pipeline import Pipeline

    artifacts = state.get("artifacts", {})

    preprocessor_path = project_dir / artifacts.get("preprocessor", "artifacts/preprocessor.joblib")
    model_key = "model_tuned" if "model_tuned" in artifacts else "model"
    model_path = project_dir / artifacts.get(model_key, f"artifacts/{model_key}.joblib")

    try:
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
    except Exception as e:
        console.print(f"[red]Could not load preprocessor/model: {e}[/red]")
        return None

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    out_path = project_dir / "artifacts" / "pipeline.joblib"
    joblib.dump(pipeline, out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Pipeline script
# ---------------------------------------------------------------------------

def _export_pipeline_script(state: dict, project_dir: Path) -> str | None:
    fp = state.get("feature_preparation", {})
    fs = state.get("feature_selection", {})
    mod = state.get("modeling", {})
    tun = state.get("tuning", {})

    project_name = state["project"]["name"]
    target_col = fp.get("target_column", "target")
    problem_type = fp.get("problem_type", "classification")
    encoding = fp.get("encoding", {})
    scaling = fp.get("scaling", {})
    selected_features = fs.get("selected_features", [])

    params = tun.get("best_params") or mod.get("params", {})
    model_class = mod.get("model_class", "sklearn.ensemble.RandomForestClassifier")
    model_module, model_name = model_class.rsplit(".", 1)

    one_hot_cols = [c for c, m in encoding.items() if m["strategy"] == "one_hot"]
    ordinal_cols = [c for c, m in encoding.items() if m["strategy"] == "ordinal"]
    standard_cols = [c for c, m in scaling.items() if m["strategy"] == "standard"]
    minmax_cols = [c for c, m in scaling.items() if m["strategy"] == "minmax"]

    imports = {"from sklearn.pipeline import Pipeline", "from sklearn.compose import ColumnTransformer"}
    if one_hot_cols:
        imports.add("from sklearn.preprocessing import OneHotEncoder")
    if ordinal_cols:
        imports.add("from sklearn.preprocessing import OrdinalEncoder")
    if standard_cols:
        imports.add("from sklearn.preprocessing import StandardScaler")
    if minmax_cols:
        imports.add("from sklearn.preprocessing import MinMaxScaler")
    imports.add(f"from {model_module} import {model_name}")

    params_str = _format_params(params)
    transformers_str = _format_transformers(one_hot_cols, ordinal_cols, standard_cols, minmax_cols)

    lines = [
        "# ============================================================",
        "# ML Orchestrator — Generated Pipeline Script",
        f"# Project : {project_name}",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Problem  : {problem_type}",
        f"# Target   : {target_col}",
        "# ============================================================",
        "",
    ]
    lines += sorted(imports)
    lines += [
        "",
        "",
        "# --- Feature list (after selection) ---",
        f"SELECTED_FEATURES = {selected_features!r}",
        "",
        "",
        "# --- Preprocessing ---",
        "# Each transformer is applied to its column group.",
        "# 'remainder=passthrough' keeps any unspecified columns as-is.",
        "preprocessor = ColumnTransformer(",
        "    transformers=[",
    ]
    for t_line in transformers_str:
        lines.append(f"        {t_line}")
    lines += [
        "    ],",
        "    remainder='passthrough',",
        ")",
        "",
        "",
        f"# --- Model: {model_name} ---",
        f"model = {model_name}(",
    ]
    for p_line in params_str:
        lines.append(f"    {p_line}")
    lines += [
        ")",
        "",
        "",
        "# --- Full Pipeline ---",
        "# Call pipeline.fit(X_train, y_train) to train.",
        "# Call pipeline.predict(X_new) to score new data.",
        "# X must contain SELECTED_FEATURES columns (before preprocessing).",
        "pipeline = Pipeline([",
        "    ('preprocessor', preprocessor),",
        "    ('model', model),",
        "])",
    ]

    out_path = project_dir / "artifacts" / "pipeline_script.py"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)


def _format_params(params: dict) -> list[str]:
    if not params:
        return []
    lines = []
    for k, v in params.items():
        lines.append(f"{k}={v!r},")
    return lines


def _format_transformers(one_hot, ordinal, standard, minmax) -> list[str]:
    lines = []
    if one_hot:
        lines.append(f"('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), {one_hot!r}),")
    if ordinal:
        lines.append(f"('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), {ordinal!r}),")
    if standard:
        lines.append(f"('standard', StandardScaler(), {standard!r}),")
    if minmax:
        lines.append(f"('minmax', MinMaxScaler(), {minmax!r}),")
    if not lines:
        lines.append("# No transformers configured")
    return lines


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _export_run_summary(state: dict, project_dir: Path) -> str | None:
    proj = state.get("project", {})
    ing = state.get("ingestion", {})
    prof = state.get("profiling", {})
    cln = state.get("cleaning", {})
    fp = state.get("feature_preparation", {})
    fs = state.get("feature_selection", {})
    mod = state.get("modeling", {})
    tun = state.get("tuning", {})
    ev = state.get("evaluation", {})

    lines = [
        "# ML Orchestrator — Run Summary",
        "",
        f"**Project:** {proj.get('name', '—')}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "---",
        "",
        "## Dataset",
        f"- Source: `{ing.get('source_path', '—')}`",
        f"- Format: {ing.get('format', '—')}",
        f"- Rows: {ing.get('row_count', '—'):,}" if isinstance(ing.get('row_count'), int) else f"- Rows: {ing.get('row_count', '—')}",
        f"- Columns: {ing.get('column_count', '—')}",
        "",
    ]

    if prof:
        lines += [
            "## Profiling",
            f"- Duplicate rows: {prof.get('duplicate_rows', 0)}",
            f"- Warnings: {len(prof.get('warnings', []))}",
            "",
        ]

    if cln:
        actions = cln.get("actions", [])
        excluded = cln.get("excluded_columns", [])
        lines += [
            "## Cleaning",
            f"- Actions applied: {len(actions)}",
            f"- Excluded columns: {', '.join(excluded) if excluded else 'none'}",
            f"- Rows after cleaning: {cln.get('row_count', '—')}",
            "",
        ]

    if fp:
        lines += [
            "## Feature Preparation",
            f"- Target column: `{fp.get('target_column', '—')}`",
            f"- Problem type: {fp.get('problem_type', '—')} ({fp.get('problem_type_source', '')})",
            f"- Train rows: {fp.get('split', {}).get('train_rows', '—')}",
            f"- Test rows: {fp.get('split', {}).get('test_rows', '—')}",
            f"- Features after encoding: {fp.get('feature_count', '—')}",
            "",
        ]

    if fs:
        lines += [
            "## Feature Selection",
            f"- Method: {fs.get('method', '—')}",
            f"- Threshold: {fs.get('method_params', {}).get('threshold', '—')}",
            f"- Selected: {len(fs.get('selected_features', []))} features",
            f"- Removed: {len(fs.get('removed_features', []))} features",
            "",
        ]

    if mod:
        params = tun.get("best_params") or mod.get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "defaults"
        lines += [
            "## Model",
            f"- Algorithm: {mod.get('model_name', '—')}",
            f"- Parameters: {params_str}",
            f"- Tuned: {'yes' if tun.get('retrained') else 'no'}",
            "",
        ]

    if ev:
        metrics = ev.get("metrics", {})
        lines += [
            "## Evaluation",
            f"- Model used: {ev.get('model_used', '—')}",
        ]
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    out_path = project_dir / "artifacts" / "run_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------

def _export_requirements(project_dir: Path) -> str | None:
    packages = [
        "pandas",
        "scikit-learn",
        "joblib",
        "rich",
        "questionary",
        "plotext",
        "typer",
        "pyyaml",
    ]
    out_path = project_dir / "artifacts" / "requirements.txt"
    out_path.write_text("\n".join(packages) + "\n", encoding="utf-8")
    return str(out_path)
