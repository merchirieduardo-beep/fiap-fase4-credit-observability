"""Etapa 3 — Camada de observabilidade: métricas, alertas, MLflow e Prometheus."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .config import (ALERTS_FILE, AUC_DROP_ALERT, CALIBRATION_GAP_ALERT,
                     DISPARATE_IMPACT_MIN, DRIFT_SHARE_ALERT, FAIRNESS_DEGRADATION, METRICS_FILE, MONITORING_DIR,
                     PSI_DRIFT, ROOT)
from .logging_utils import get_logger, log_event

logger = get_logger("observability")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def reset_history() -> None:
    for f in (METRICS_FILE, ALERTS_FILE):
        f.write_text("", encoding="utf-8")


def append_jsonl(path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def alert(batch: str, severity: str, kind: str, message: str, value=None, threshold=None) -> dict:
    record = {"ts": now(), "batch": batch, "severity": severity, "type": kind,
              "message": message, "value": value, "threshold": threshold}
    append_jsonl(ALERTS_FILE, record)
    log_event(logger, "alerta", "error" if severity == "critico" else "warning",
              lote=batch, tipo=kind, severidade=severity, mensagem=message)
    return record


def evaluate_alerts(rec: dict, card: dict) -> list[dict]:
    """Regras de alerta aplicadas a cada lote monitorado."""
    batch, alerts = rec["batch"], []
    baseline, base_fair = card["baseline_metrics"], card.get("baseline_fairness", {})
    for feat, psi in rec["psi"].items():
        if psi >= PSI_DRIFT:
            alerts.append(alert(batch, "critico", "data_drift",
                                f"Data drift em '{feat}' (PSI={psi:.3f})", round(psi, 4), PSI_DRIFT))
    if rec["score_psi"] >= PSI_DRIFT:
        alerts.append(alert(batch, "critico", "prediction_drift",
                            f"Distribuição do score mudou (PSI={rec['score_psi']:.3f})",
                            round(rec["score_psi"], 4), PSI_DRIFT))
    if rec["drift_share"] >= DRIFT_SHARE_ALERT:
        alerts.append(alert(batch, "critico", "retrain_recommended",
                            f"{rec['n_drifted_features']} features com drift "
                            f"({rec['drift_share']:.0%}) — acionar runbook de retreino",
                            round(rec["drift_share"], 4), DRIFT_SHARE_ALERT))
    perf = rec.get("performance")
    if perf:
        drop = baseline["auc"] - perf["auc"]
        if drop >= AUC_DROP_ALERT:
            alerts.append(alert(batch, "critico", "performance_degradation",
                                f"AUC caiu {drop:.3f} vs. baseline ({baseline['auc']:.3f} -> {perf['auc']:.3f})",
                                round(perf["auc"], 4), round(baseline["auc"] - AUC_DROP_ALERT, 4)))
        # modelo subestimando o risco: a relação X -> y mudou (data drift sozinho não causa isso)
        gap = perf["default_rate"] - perf["mean_score"]
        if gap >= CALIBRATION_GAP_ALERT:
            alerts.append(alert(batch, "critico", "concept_drift",
                                f"Modelo subestima o risco: inadimplência real {perf['default_rate']:.1%} "
                                f"vs. {perf['mean_score']:.1%} prevista (gap={gap:.1%})",
                                round(gap, 4), CALIBRATION_GAP_ALERT))
    for group, info in rec.get("fairness", {}).items():
        base_di = base_fair.get(group, {}).get("disparate_impact", 1.0)
        # alerta quando viola a regra dos 4/5 E piorou em relação ao baseline documentado
        if info["disparate_impact"] < DISPARATE_IMPACT_MIN and info["disparate_impact"] < base_di - FAIRNESS_DEGRADATION:
            alerts.append(alert(batch, "atencao", "fairness",
                                f"Disparidade de aprovação por {group} piorou "
                                f"(razão={info['disparate_impact']:.2f}; baseline={base_di:.2f})",
                                info["disparate_impact"], DISPARATE_IMPACT_MIN))
    return alerts


# ------------------------------------------------------------------ integrações opcionais
def log_to_mlflow(rec: dict, artifacts: list | None = None) -> None:
    """Rastreia cada lote como um run no MLflow (experimento credit-monitoring)."""
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
        mlflow.set_experiment("credit-monitoring")
        with mlflow.start_run(run_name=f"batch_{rec['batch']}"):
            mlflow.set_tags({"batch": rec["batch"], "contract_status": rec["contract_status"]})
            flat = {
                "rows_received": rec["rows_received"], "rows_ingested": rec["rows_ingested"],
                "contract_failures": rec["contract_failures"],
                "latency_validation_ms": rec["latency_validation_ms"],
                "alerts": rec["n_alerts"],
            }
            if rec["contract_status"] == "aprovado":
                flat.update({"latency_scoring_ms": rec["latency_scoring_ms"],
                             "n_drifted_features": rec["n_drifted_features"],
                             "drift_share": rec["drift_share"], "score_psi": rec["score_psi"],
                             "approval_rate": rec["approval_rate"]})
                flat.update({f"psi_{k}": v for k, v in rec["psi"].items()})
                flat.update({f"perf_{k}": v for k, v in (rec.get("performance") or {}).items()})
            mlflow.log_metrics({k: float(v) for k, v in flat.items()})
            for path in artifacts or []:
                mlflow.log_artifact(str(path))
    except Exception as exc:  # observabilidade nunca derruba o pipeline
        log_event(logger, "mlflow_falhou", "warning", erro=type(exc).__name__)


def export_prometheus(history: list[dict]) -> None:
    """Exporta o estado do último lote em formato Prometheus (textfile collector)."""
    try:
        from prometheus_client import CollectorRegistry, Gauge, write_to_textfile
    except ImportError:
        return
    ok = [r for r in history if r["contract_status"] == "aprovado"]
    if not ok:
        return
    last, registry = ok[-1], CollectorRegistry()
    psi = Gauge("credit_feature_psi", "PSI por feature (último lote)", ["feature"], registry=registry)
    for feat, value in last["psi"].items():
        psi.labels(feature=feat).set(value)
    Gauge("credit_score_psi", "PSI do score", registry=registry).set(last["score_psi"])
    Gauge("credit_drifted_features", "Features com drift", registry=registry).set(last["n_drifted_features"])
    Gauge("credit_rows_ingested", "Linhas ingeridas", registry=registry).set(last["rows_ingested"])
    Gauge("credit_scoring_latency_ms", "Latência de scoring (ms)", registry=registry).set(last["latency_scoring_ms"])
    Gauge("credit_batches_blocked_total", "Lotes bloqueados pelo contrato", registry=registry).set(
        sum(r["contract_status"] == "bloqueado" for r in history))
    Gauge("credit_alerts_total", "Alertas emitidos", registry=registry).set(
        sum(r["n_alerts"] for r in history))
    if last.get("performance"):
        Gauge("credit_model_auc", "AUC com rótulos maduros", registry=registry).set(last["performance"]["auc"])
    write_to_textfile(str(MONITORING_DIR / "metrics.prom"), registry)
