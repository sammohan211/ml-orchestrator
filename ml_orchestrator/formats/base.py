from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseFormat(ABC):
    """Interface that every file format handler must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable format name shown in menus. E.g. 'CSV'."""

    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """File extensions this format handles. E.g. ['.csv']."""

    @property
    @abstractmethod
    def required_packages(self) -> list[str]:
        """Pip packages required to load this format. E.g. ['pyarrow']."""

    @abstractmethod
    def load(self, path: Path) -> pd.DataFrame:
        """Load the file at path and return a DataFrame."""
