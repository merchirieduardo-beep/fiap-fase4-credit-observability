"""Etapa 1 — Gera o Dataset de Referência, valida no contrato e treina o baseline.

Uso: python -m src.train
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split

from .config import (FEATURES, N_REFERENCE, RAW_DIR, REFERENCE_DIR, REFERENCE_SNAPSHOT,
                     REPORTS_DIR, SEED, TARGET)
from .contracts import CONTRACT_VERSION, validate_batch
from .datagen import generate_raw
from .fairness import approval_by_group
from .logging_utils import get_logger, log_event
from .model import (MODEL_CARD_PATH, MODEL_PATH, best_threshold, build_pipeline,
                    classification_metrics, score_frame)
from .privacy import apply_privacy

logger = get_logger("train")


def main() -> dict:
    # 1) dado bruto (com PII sintética) -> camada de privacidade -> dado analítico
    raw = generate_raw(N_REFERENCE, SEED, REFERENCE_SNAPSHOT)
    raw.to_csv(RAW_DIR / "reference_raw.csv", index=False)  # fora do versionamento
    reference = apply_privacy(raw, REFERENCE_SNAPSHOT)

    # 2) contrato de dados: só treina com dado aprovado
    reference = validate_batch(reference, "reference", with_target=True)
    reference.to_csv(REFERENCE_DIR / "reference.csv", index=False)

    # 3) treino do baseline
    train, holdout = train_test_split(reference, test_size=0.2, random_state=SEED,
                                      stratify=reference[TARGET])
    model, algo = build_pipeline()
    model.fit(train[FEATURES], train[TARGET])

    holdout_scores = model.predict_proba(holdout[FEATURES])[:, 1]
    threshold = best_threshold(holdout[TARGET], holdout_scores)
    metrics = classification_metrics(holdout[TARGET], holdout_scores, threshold)

    # 4) artefatos: modelo, model card e referência pontuada (base das comparações de drift)
    joblib.dump(model, MODEL_PATH)
    scored_holdout = score_frame(model, holdout, threshold)
    scored_holdout.to_csv(REFERENCE_DIR / "reference_scored.csv", index=False)
    card = {
        "model_name": "credit_scoring_baseline",
        "algorithm": algo,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_snapshot": REFERENCE_SNAPSHOT,
        "data_contract_version": CONTRACT_VERSION,
        "reference_sha256": hashlib.sha256(
            (REFERENCE_DIR / "reference.csv").read_bytes()).hexdigest(),
        "n_train": len(train), "n_holdout": len(holdout),
        "features": FEATURES,
        "excluded_from_model": ["gender (atributo protegido, só auditoria)", "customer_hash"],
        "decision_threshold": threshold,
        "decision_rule": "score >= threshold -> negar / encaminhar para revisão humana",
        "baseline_metrics": metrics,
        # viés já existente no baseline: referência para detectar piora em produção
        "baseline_fairness": approval_by_group(scored_holdout),
    }
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "model_card.json").write_text(  # cópia versionada (evidência de governança)
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event(logger, "modelo_treinado", algoritmo=algo, auc=round(metrics["auc"], 4),
              ks=round(metrics["ks"], 4), threshold=round(threshold, 4), n_train=len(train))
    return card


if __name__ == "__main__":
    main()
