from pathlib import Path

import pandas as pd

from ml_orchestrator.formats.base import BaseFormat


class CSVFormat(BaseFormat):

    @property
    def name(self) -> str:
        return "CSV"

    @property
    def extensions(self) -> list[str]:
        return [".csv"]

    @property
    def required_packages(self) -> list[str]:
        return []  # pandas is always present

    def load(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)
