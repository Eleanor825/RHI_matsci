from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_learned_live_matbot_harness.py")
SPEC = importlib.util.spec_from_file_location("_frozen_live_matbot_evaluator", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load frozen live MatBot evaluator")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ORIGINAL_LOGISTIC = MODULE.LogisticRegression


class DegenerateSafeLogisticRegression:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._delegate = None
        self._constant = None

    def fit(self, features, labels):
        labels = np.asarray(labels, dtype=int)
        if len(np.unique(labels)) == 1:
            self._constant = float((labels.sum() + 1.0) / (len(labels) + 2.0))
            return self
        self._delegate = ORIGINAL_LOGISTIC(*self._args, **self._kwargs).fit(features, labels)
        return self

    def predict_proba(self, features):
        if self._delegate is not None:
            return self._delegate.predict_proba(features)
        count = len(features)
        positive = np.full(count, self._constant, dtype=float)
        return np.column_stack((1.0 - positive, positive))


MODULE.LogisticRegression = DegenerateSafeLogisticRegression


if __name__ == "__main__":
    raise SystemExit(MODULE.main())
