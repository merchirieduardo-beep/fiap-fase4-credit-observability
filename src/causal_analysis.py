"""Análise causal da degradação — por que o modelo piorou?

Como o processo gerador é conhecido, dá para fazer INTERVENÇÕES (do-operator) em vez de
só observar correlações: liga-se um mecanismo por vez, mantendo o resto no estado de
referência, e mede-se o efeito sobre o modelo congelado.

  Cenário A  — baseline (sem drift)
  Cenário B  — só data drift       : muda P(X), mantém P(y|X)
  Cenário C  — só concept drift    : mantém P(X), muda P(y|X)
  Cenário D  — ambos (produção real no último mês)
  + ablação de cada mecanismo individual

Uso: python -m src.causal_analysis
"""
from __future__ import annotations

import json

import pandas as pd

from .config import REPORTS_DIR, SEED
from .datagen import CONCEPT_DRIFT_KEYS, DATA_DRIFT_KEYS, generate_raw, uniform
from .logging_utils import get_logger, log_event
from .model import classification_metrics, load_model
from .config import FEATURES, TARGET

logger = get_logger("causal")
N = 20_000
SNAPSHOT = "2026-10-31"

LABELS = {
    "income": "Data: renda nominal inflacionada",
    "unemployment": "Data: desemprego 6% → 20%",
    "utilization": "Data: maior uso do limite de crédito",
    "age": "Data: novo perfil (clientes mais jovens)",
    "late_payments": "Data: mais atrasos recentes",
    "real_income": "Conceito: renda real corroída pela inflação",
    "self_employed_risk": "Conceito: autônomos ficaram mais arriscados",
    "utilization_weight": "Conceito: uso do limite pesa mais no risco",
    "history_signal_decay": "Conceito: histórico (DTI/atrasos) perdeu poder preditivo",
}


def _evaluate(model, threshold, data=None, concept=None) -> dict:
    raw = generate_raw(N, SEED + 999, SNAPSHOT, data_drift=data, concept_drift=concept)
    raw["age"] = ((pd.Timestamp(SNAPSHOT) - pd.to_datetime(raw["birth_date"])).dt.days // 365.25).astype(int)
    scores = model.predict_proba(raw[FEATURES])[:, 1]
    return classification_metrics(raw[TARGET], scores, threshold)


def main() -> dict:
    model, card = load_model()
    thr = card["decision_threshold"]
    scenarios = {
        "A. Baseline (sem drift)": _evaluate(model, thr),
        "B. Só data drift — P(X) muda": _evaluate(model, thr, data=uniform(DATA_DRIFT_KEYS, 1)),
        "C. Só concept drift — P(y|X) muda": _evaluate(model, thr, concept=uniform(CONCEPT_DRIFT_KEYS, 1)),
        "D. Data + concept drift": _evaluate(model, thr, data=uniform(DATA_DRIFT_KEYS, 1),
                                             concept=uniform(CONCEPT_DRIFT_KEYS, 1)),
    }
    base = scenarios["A. Baseline (sem drift)"]
    ablation = []
    for key in DATA_DRIFT_KEYS:
        m = _evaluate(model, thr, data={key: 1.0})
        ablation.append({"mecanismo": LABELS[key], "tipo": "data", **m})
    for key in CONCEPT_DRIFT_KEYS:
        m = _evaluate(model, thr, concept={key: 1.0})
        ablation.append({"mecanismo": LABELS[key], "tipo": "conceito", **m})
    for row in ablation:
        row["delta_auc"] = row["auc"] - base["auc"]
        row["delta_bad_rate_approved"] = row["bad_rate_approved"] - base["bad_rate_approved"]
        # erro de calibração: quanto o modelo SUBESTIMA o risco (real - previsto)
        row["risco_subestimado"] = row["default_rate"] - row["mean_score"]
    ablation.sort(key=lambda r: r["delta_auc"])

    result = {"n_por_cenario": N, "scenarios": scenarios, "ablation": ablation}
    (REPORTS_DIR / "causal_analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Análise causal da degradação (gerado por `python -m src.causal_analysis`)", "",
             "## Decomposição: data drift vs. concept drift", "",
             "| Cenário | AUC | Inadimplência real | Score médio previsto | Aprovação | Inadimplência entre aprovados |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in scenarios.items():
        lines.append(f"| {name} | {m['auc']:.3f} | {m['default_rate']:.1%} | {m['mean_score']:.1%} "
                     f"| {m['approval_rate']:.1%} | {m['bad_rate_approved']:.1%} |")
    lines += ["", "## Ablação: efeito de cada mecanismo isolado (intervenção única)", "",
              "| Mecanismo | ΔAUC | Δ inadimplência entre aprovados | Risco subestimado (real − previsto) |",
              "|---|---:|---:|---:|"]
    for r in ablation:
        lines.append(f"| {r['mecanismo']} | {r['delta_auc']:+.3f} | "
                     f"{r['delta_bad_rate_approved']:+.1%} | {r['risco_subestimado']:+.1%} |")
    (REPORTS_DIR / "causal_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_event(logger, "analise_causal_concluida",
              auc_baseline=round(base["auc"], 3),
              auc_so_data_drift=round(scenarios["B. Só data drift — P(X) muda"]["auc"], 3),
              auc_so_concept_drift=round(scenarios["C. Só concept drift — P(y|X) muda"]["auc"], 3))
    return result


if __name__ == "__main__":
    main()
