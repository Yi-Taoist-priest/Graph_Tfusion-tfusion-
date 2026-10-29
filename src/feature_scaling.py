"""Leakage-safe robust scaling for host and graph context features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class RobustFeatureScaler:
    feature_names: Tuple[str, ...]

    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = tuple(str(name) for name in feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty.")
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _rows(self, rows, *, require_non_empty: bool) -> np.ndarray:
        array = np.asarray(rows, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Feature rows must be a two-dimensional matrix.")
        if require_non_empty and array.shape[0] == 0:
            raise ValueError("Feature rows must be non-empty.")
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature width {array.shape[1]} does not match schema width "
                f"{len(self.feature_names)}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Feature rows must contain only finite values.")
        return array

    def fit(self, rows) -> "RobustFeatureScaler":
        array = self._rows(rows, require_non_empty=True)
        center = np.median(array, axis=0)
        q1, q3 = np.quantile(array, [0.25, 0.75], axis=0)
        spread = q3 - q1
        self.center_ = np.asarray(center, dtype=np.float32)
        self.scale_ = np.asarray(np.where(spread > 0.0, spread, 1.0), dtype=np.float32)
        return self

    def transform(self, rows) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustFeatureScaler must be fitted before transform().")
        array = self._rows(rows, require_non_empty=False)
        return np.asarray((array - self.center_) / self.scale_, dtype=np.float32)

    def fit_transform(self, rows) -> np.ndarray:
        return self.fit(rows).transform(rows)

    def to_dict(self) -> dict:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustFeatureScaler must be fitted before serialization.")
        return {
            "feature_names": list(self.feature_names),
            "center": [float(value) for value in self.center_],
            "scale": [float(value) for value in self.scale_],
        }
