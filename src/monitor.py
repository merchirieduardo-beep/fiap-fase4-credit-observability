"""Etapas 2 e 3 — Pipeline de monitoramento dos lotes de produção.

Para cada lote em data/production (em ordem cronológica):
  ingestão -> contrato de dados -> (bloqueia | scoring) -> drift -> performance -> viés -> alertas

Uso: python -m src.monitor
"""
from __future__ import annotations

import time

import pandas as pd

from .config import (CAT_FEATURES, FEATURES, ID_COLUMN, METRICS_FILE, NUM_FEATURES,
                     PRODUCTION_DIR, REFERENCE_DIR, REPORTS_DIR, SCORED_DIR, TARGET)
from .contracts import DataContractViolation, quarantine, validate_batch
from .drift import compute_drift, evidently_reports
from .fairness import approval_by_group
from .logging_utils import get_logger, log_event
from .model import classification_metrics, load_model, score_frame
from .observability import (alert, append_jsonl, evaluate_alerts, export_prometheus,
                            log_to_mlflow, now, read_jsonl, reset_history)

logger = get_logger("monitor")
DRIFT_DIR = REPORTS_DIR / "drift"


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def process_batch(path, model, card, reference: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    name = path.stem.replace("batch_", "")
    log_event(logger, "ingestao_iniciada", lote=name, arquivo=path.name)
    t0 = time.perf_counter()
    df = pd.read_csv(path, dtype={ID_COLUMN: str})
    rec = {"batch": name, "ts": now(), "rows_received": len(df), "rows_ingested": 0,
           "contract_failures": 0, "n_alerts": 0}

    # ---- 1. contrato de dados
    t = time.perf_counter()
    try:
        df = validate_batch(df, name)
    except DataContractViolation as violation:
        quarantine(df, violation)
        alert(name, "critico", "data_contract",
              f"Lote bloqueado: {len(violation.failure_cases)} violações em "
              f"{len(violation.summary)} regras do contrato", len(violation.failure_cases), 0)
        rec.update(contract_status="bloqueado", contract_failures=len(violation.failure_cases),
                   rules_violated=violation.summary.head(10).astype(str).to_dict("records"),
                   latency_validation_ms=_ms(t), latency_total_ms=_ms(t0), n_alerts=1,
                   status="falha")
        return rec, None
    rec.update(contract_status="aprovado", latency_validation_ms=_ms(t), rows_ingested=len(df))

    # ---- 2. scoring
    t = time.perf_counter()
    scored = score_frame(model, df, card["decision_threshold"])
    rec["latency_scoring_ms"] = _ms(t)
    rec["latency_per_1k_rows_ms"] = round(rec["latency_scoring_ms"] / len(df) * 1000, 2)
    rec["approval_rate"] = float((scored["prediction"] == 0).mean())
    rec["mean_score"] = float(scored["score"].mean())

    # ---- 3. drift (features + score)
    table = compute_drift(reference, scored, NUM_FEATURES + ["score"], CAT_FEATURES)
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(DRIFT_DIR / f"drift_{name}.csv", index=False)
    feats = table[table["feature"].isin(FEATURES)]
    rec["psi"] = {r.feature: round(r.psi, 4) for r in feats.itertuples()}
    rec["ks_pvalue"] = {r.feature: float(f"{r.p_valor:.3g}") for r in feats.itertuples()}
    rec["drifted_features"] = feats.loc[feats["status"] == "drift", "feature"].tolist()
    rec["n_drifted_features"] = len(rec["drifted_features"])
    rec["drift_share"] = round(rec["n_drifted_features"] / len(FEATURES), 4)
    rec["score_psi"] = round(float(table.loc[table["feature"] == "score", "psi"].iloc[0]), 4)

    # ---- 4. performance com rótulos maduros (quando já disponíveis)
    labels_path = path.with_name(f"labels_{name}.csv")
    if labels_path.exists():
        labels = pd.read_csv(labels_path, dtype={ID_COLUMN: str})
        scored = scored.merge(labels, on=ID_COLUMN, how="left", validate="one_to_one")
        rec["performance"] = classification_metrics(scored[TARGET], scored["score"],
                                                    card["decision_threshold"])

    # ---- 5. auditoria de viés e alertas
    rec["fairness"] = approval_by_group(scored)
    rec["n_alerts"] = len(evaluate_alerts(rec, card))
    rec["latency_total_ms"] = _ms(t0)
    rec["status"] = "sucesso"
    scored.to_csv(SCORED_DIR / f"scored_{name}.csv", index=False)
    log_event(logger, "lote_processado", lote=name, linhas=len(df),
              features_com_drift=rec["n_drifted_features"], alertas=rec["n_alerts"],
              latencia_ms=rec["latency_total_ms"])
    return rec, scored


def main() -> list[dict]:
    model, card = load_model()
    reference = pd.read_csv(REFERENCE_DIR / "reference_scored.csv", dtype={ID_COLUMN: str})
    reset_history()
    last_scored, last_name = None, None

    for path in sorted(PRODUCTION_DIR.glob("batch_*.csv")):
        try:
            rec, scored = process_batch(path, model, card, reference)
        except Exception as exc:  # falha inesperada: registra e segue para o próximo lote
            name = path.stem.replace("batch_", "")
            log_event(logger, "falha_no_pipeline", "error", lote=name, erro=repr(exc))
            alert(name, "critico", "pipeline_failure", f"Falha inesperada: {type(exc).__name__}")
            rec, scored = {"batch": name, "ts": now(), "status": "falha", "contract_status": "erro",
                           "rows_received": 0, "rows_ingested": 0, "contract_failures": 0,
                           "latency_validation_ms": 0, "n_alerts": 1}, None
        append_jsonl(METRICS_FILE, rec)
        log_to_mlflow(rec, [DRIFT_DIR / f"drift_{rec['batch']}.csv"] if scored is not None else None)
        if scored is not None:
            last_scored, last_name = scored, rec["batch"]

    history = read_jsonl(METRICS_FILE)
    # visão consolidada: PSI por feature x lote
    ok = [r for r in history if r["contract_status"] == "aprovado"]
    summary = pd.DataFrame({r["batch"]: {**r["psi"], "score (predição)": r["score_psi"]} for r in ok})
    summary.to_csv(DRIFT_DIR / "psi_por_lote.csv", index_label="feature")

    # relatório Evidently: Referência vs. lote mais recente (pior caso)
    if last_scored is not None:
        try:
            evidently_reports(reference, last_scored, REPORTS_DIR / "evidently", last_name, target=TARGET)
        except Exception as exc:  # relatório é evidência, não pode derrubar o monitoramento
            log_event(logger, "evidently_falhou", "error", erro=repr(exc)[:300])
    export_prometheus(history)
    log_event(logger, "monitoramento_concluido", lotes=len(history),
              bloqueados=sum(r["contract_status"] == "bloqueado" for r in history),
              alertas=sum(r["n_alerts"] for r in history))
    return history


if __name__ == "__main__":
    main()
