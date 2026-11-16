"""Modelo de risco de crédito: construção, métricas e scoring."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CAT_FEATURES, FEATURES, MODELS_DIR, NUM_FEATURES, SEED

MODEL_PATH = MODELS_DIR / "credit_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"


def build_pipeline() -> tuple[Pipeline, str]:
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, eval_metric="logloss", random_state=SEED)
        algo = "XGBoost"
    except ImportError:  # fallback para ambientes sem xgboost
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                             random_state=SEED)
        algo = "HistGradientBoosting (scikit-learn)"
    return Pipeline([("pre", pre), ("clf", clf)]), algo


def ks_statistic(y_true, scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def best_threshold(y_true, scores) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


def classification_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    approved = pred == 0
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ks": ks_statistic(y_true, scores),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, scores)),
        "default_rate": float(y_true.mean()),
        "mean_score": float(scores.mean()),
        "approval_rate": float(approved.mean()),
        # inadimplência entre os APROVADOS = perda efetiva da carteira
        "bad_rate_approved": float(y_true[approved].mean()) if approved.any() else 0.0,
    }


def load_model() -> tuple[Pipeline, dict]:
    return joblib.load(MODEL_PATH), json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))


def score_frame(model: Pipeline, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Adiciona `score` (P(default)) e `prediction` (1 = negar crédito)."""
    out = df.copy()
    out["score"] = model.predict_proba(df[FEATURES])[:, 1]
    out["prediction"] = (out["score"] >= threshold).astype(int)
    return out
