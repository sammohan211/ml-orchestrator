from pathlib import Path

import pandas as pd

from ml_orchestrator.formats.base import BaseFormat


class ParquetFormat(BaseFormat):

    @property
    def name(self) -> str:
        return "Parquet"

    @property
    def extensions(self) -> list[str]:
        return [".parquet"]

    @property
    def required_packages(self) -> list[str]:
        return ["pyarrow"]

    def load(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)
