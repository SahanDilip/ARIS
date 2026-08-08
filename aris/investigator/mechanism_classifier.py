"""Mechanism classifier: recovers oracle POLICY FAMILY from observed behaviour.

Takes precomputed summary-statistic feature matrices (see
investigator_driver.py::collect_training_replicates, which does the platform-
facing data collection -- this module MUST NOT import aris.platform, enforced
by tests/test_barrier.py, so all platform interaction lives in the driver).

The features themselves are aris/diagnostics.py's summary_statistics, unchanged
-- the same ones the Step-1 separability diagnostic used to establish that
family classification saturates quickly (see STEP1_REPORT.md 4a). This module
turns that diagnostic into an actual trained classifier reporting held-out
accuracy per family, which is the paper's main recovery table
(04_EVALUATION.md, E1).

Given Step 1's finding, this is expected to score well; the real headroom is in
Step 4's parameter-level recovery, not here. Reporting a high, saturated
accuracy honestly IS a result, not a null outcome (00_RESEARCH_DIRECTION.md,
Defence 2).
"""
from __future__ import annotations

import numpy as np

from ..diagnostics import FEATURE_NAMES


def fit_and_evaluate(feats_by_oracle: dict, seed: int, test_frac: float = 0.3) -> dict:
    """Train/test split within each family, fit a multiclass classifier, report
    held-out accuracy, macro-F1, and the confusion matrix (04_EVALUATION.md's
    "Policy-family classification accuracy / macro-F1" metric).

    `feats_by_oracle` is {oracle_name: (n_replicates, n_features) array}.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    names = sorted(feats_by_oracle)
    X = np.vstack([feats_by_oracle[o] for o in names])
    y = np.concatenate([[o] * len(feats_by_oracle[o]) for o in names])

    if len(X) < 2 * len(names):
        return {"fitted": False, "reason": "too few replicates for a train/test split"}

    counts = np.unique(y, return_counts=True)[1]
    strat = y if counts.min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, random_state=seed, stratify=strat)
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=seed))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float((pred == yte).mean())
    f1 = float(f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred, labels=names)

    importances = None
    try:
        rf = clf.named_steps["randomforestclassifier"]
        importances = {FEATURE_NAMES[i]: round(float(v), 4)
                      for i, v in enumerate(rf.feature_importances_)}
    except Exception:
        pass

    return {
        "fitted": True, "n_train": len(Xtr), "n_test": len(Xte),
        "accuracy": round(acc, 3), "macro_f1": round(f1, 3),
        "families": names, "confusion_matrix": cm.tolist(),
        "feature_importances": importances,
    }