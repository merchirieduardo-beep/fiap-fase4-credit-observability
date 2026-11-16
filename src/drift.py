"""Etapa 2 — Detecção estatística de drift.

Duas camadas complementares:
1. Testes próprios (PSI, Kolmogorov-Smirnov, Qui-quadrado) -> alimentam alertas e métricas.
2. Evidently AI -> relatório HTML interativo, Referência vs. Produção.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import CAT_FEATURES, KS_ALPHA, NUM_FEATURES, PSI_DRIFT, PSI_WARN
from .logging_utils import get_logger, log_event

logger = get_logger("drift")
_EPS = 1e-4


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index com faixas definidas pelos decis da REFERÊNCIA."""
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variável quase constante: trata como categórica
        return psi_categorical(ref, cur)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref.dropna(), edges)[0] / max(len(ref.dropna()), 1)
    cur_pct = np.histogram(cur.dropna(), edges)[0] / max(len(cur.dropna()), 1)
    return _psi(ref_pct, cur_pct)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()), key=str)
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_pct, cur_pct)


def _status(psi: float) -> str:
    return "drift" if psi >= PSI_DRIFT else "atencao" if psi >= PSI_WARN else "estavel"


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame,
                  numeric: list[str] | None = None,
                  categorical: list[str] | None = None) -> pd.DataFrame:
    """Tabela de drift por coluna: PSI + teste de hipótese (KS ou Qui-quadrado)."""
    numeric = NUM_FEATURES if numeric is None else numeric
    categorical = CAT_FEATURES if categorical is None else categorical
    rows = []
    for col in numeric:
        ks = stats.ks_2samp(ref[col].dropna(), cur[col].dropna())
        psi = psi_numeric(ref[col], cur[col])
        rows.append({"feature": col, "tipo": "numerica", "psi": psi, "teste": "Kolmogorov-Smirnov",
                     "estatistica": float(ks.statistic), "p_valor": float(ks.pvalue),
                     "media_ref": float(ref[col].mean()), "media_prod": float(cur[col].mean())})
    for col in categorical:
        cats = sorted(set(ref[col].dropna().unique()) | set(cur[col].dropna().unique()), key=str)
        table = np.array([ref[col].value_counts().reindex(cats, fill_value=0).to_numpy(),
                          cur[col].value_counts().reindex(cats, fill_value=0).to_numpy()])
        table = table[:, table.sum(axis=0) > 0]
        chi2, p, _, _ = stats.chi2_contingency(table)
        psi = psi_categorical(ref[col], cur[col])
        rows.append({"feature": col, "tipo": "categorica", "psi": psi, "teste": "Qui-quadrado",
                     "estatistica": float(chi2), "p_valor": float(p),
                     "media_ref": None, "media_prod": None})
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].map(_status)
    out["teste_rejeita_h0"] = out["p_valor"] < KS_ALPHA
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- Evidently
def evidently_reports(ref: pd.DataFrame, cur: pd.DataFrame, out_dir: Path, tag: str,
                      target: str | None = None) -> list[Path]:
    """Gera os relatórios HTML do Evidently (Referência vs. Produção).

    - evidently_data_drift_psi_<tag>.html : drift por feature + score, teste PSI
    - evidently_data_drift_ks_<tag>.html  : idem, com KS (numéricas) e Qui-quadrado (categóricas)
    - evidently_model_performance_<tag>.html : qualidade da classificação (se houver rótulo)
    Compatível com a API 0.4.x (fixada no requirements.txt) e com a API nova (>= 0.7).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    num = NUM_FEATURES + ["score"]
    drift_cols = num + CAT_FEATURES
    generated: list[Path] = []
    try:
        import evidently  # noqa: F401
    except ImportError:
        log_event(logger, "evidently_indisponivel", "error",
                  detalhe="pip install -r requirements.txt para gerar os relatórios do Evidently")
        return generated

    try:  # ---- API legada (evidently 0.4.x)
        from evidently import ColumnMapping
        from evidently.metric_preset import ClassificationPreset, DataDriftPreset
        from evidently.report import Report

        mapping = ColumnMapping(target=None, prediction=None,
                                numerical_features=num, categorical_features=CAT_FEATURES)
        specs = {
            "psi": DataDriftPreset(num_stattest="psi", cat_stattest="psi",
                                   stattest_threshold=PSI_DRIFT),
            "ks": DataDriftPreset(num_stattest="ks", cat_stattest="chisquare",
                                  stattest_threshold=KS_ALPHA),
        }
        for name, preset in specs.items():
            report = Report(metrics=[preset])
            report.run(reference_data=ref[drift_cols], current_data=cur[drift_cols],
                       column_mapping=mapping)
            path = out_dir / f"evidently_data_drift_{name}_{tag}.html"
            report.save_html(str(path))
            generated.append(path)

        if target and target in cur.columns and target in ref.columns:
            perf_cols = NUM_FEATURES + CAT_FEATURES + [target, "prediction"]
            perf_map = ColumnMapping(target=target, prediction="prediction",
                                     numerical_features=NUM_FEATURES,
                                     categorical_features=CAT_FEATURES)
            report = Report(metrics=[ClassificationPreset()])
            report.run(reference_data=ref[perf_cols], current_data=cur[perf_cols],
                       column_mapping=perf_map)
            path = out_dir / f"evidently_model_performance_{tag}.html"
            report.save_html(str(path))
            generated.append(path)

    except ImportError:  # ---- API nova (evidently >= 0.7)
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(current_data=cur[drift_cols],
                                                   reference_data=ref[drift_cols])
        path = out_dir / f"evidently_data_drift_{tag}.html"
        snapshot.save_html(str(path))
        generated.append(path)

    log_event(logger, "evidently_relatorios_gerados", lote=tag,
              arquivos=[p.name for p in generated])
    return generated
