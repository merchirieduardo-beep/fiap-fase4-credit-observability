"""Etapa 2 — Simula a passagem do tempo em produção (Data Drift + Concept Drift).

Gera um lote mensal por período em config.PRODUCTION_BATCHES. A intensidade t do drift
cresce linearmente de 0 (primeiro mês, igual à referência) até 1 (último mês).
As features chegam primeiro (batch_*.csv); os rótulos reais chegam depois, quando a
inadimplência "amadurece" (labels_*.csv) — como em uma operação de crédito real.

Uso: python -m src.simulate_drift
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (BATCH_SIZE, ID_COLUMN, N_REFERENCE, PRODUCTION_BATCHES, PRODUCTION_DIR,
                     RAW_DIR, SEED, TARGET)
from .datagen import CONCEPT_DRIFT_KEYS, DATA_DRIFT_KEYS, generate_raw, uniform
from .logging_utils import get_logger, log_event
from .privacy import apply_privacy

logger = get_logger("simulate_drift")
CORRUPTED_SUFFIX = "_reenvio_corrompido"


def corrupt_batch(clean: pd.DataFrame, raw: pd.DataFrame | None = None, seed: int = SEED) -> pd.DataFrame:
    """Injeta os erros clássicos de ingestão em um lote (para demonstrar o contrato)."""
    rng = np.random.default_rng(seed)
    bad = clean.copy().reset_index(drop=True)
    n = len(bad)
    pick = lambda frac: rng.choice(n, size=max(1, int(n * frac)), replace=False)  # noqa: E731
    bad.loc[pick(0.03), "monthly_income"] = np.nan          # renda nula
    bad.loc[pick(0.01), "monthly_income"] = -1500.0         # renda negativa
    bad.loc[pick(0.01), "age"] = 15                         # menor de idade
    bad.loc[pick(0.005), "age"] = 130                       # idade impossível
    bad.loc[pick(0.01), "credit_utilization"] = 1.7         # fora da faixa [0, 1]
    bad.loc[pick(0.01), "employment_status"] = "DESCONHECIDO"  # categoria fora do domínio
    bad = pd.concat([bad, bad.sample(frac=0.02, random_state=seed)], ignore_index=True)  # duplicatas
    if raw is not None:  # vazamento de PII bruta: coluna fora do contrato
        ids = raw["customer_id"].reset_index(drop=True)
        bad["customer_id"] = ids.reindex(range(len(bad))).ffill()
    return bad


def main() -> list[dict]:
    manifest = []
    last = len(PRODUCTION_BATCHES) - 1
    for i, period in enumerate(PRODUCTION_BATCHES):
        t = i / last
        snapshot = (pd.Period(period, freq="M").end_time).strftime("%Y-%m-%d")
        raw = generate_raw(
            BATCH_SIZE, SEED + 100 + i, snapshot, id_offset=N_REFERENCE + i * BATCH_SIZE,
            data_drift=uniform(DATA_DRIFT_KEYS, t),
            concept_drift=uniform(CONCEPT_DRIFT_KEYS, t),
        )
        raw.to_csv(RAW_DIR / f"batch_{period}_raw.csv", index=False)
        batch = apply_privacy(raw, snapshot)

        batch.drop(columns=[TARGET]).to_csv(PRODUCTION_DIR / f"batch_{period}.csv", index=False)
        batch[[ID_COLUMN, TARGET]].to_csv(PRODUCTION_DIR / f"labels_{period}.csv", index=False)
        manifest.append({"batch": period, "drift_intensity": round(t, 2), "rows": len(batch)})
        log_event(logger, "lote_simulado", lote=period, intensidade_drift=round(t, 2), linhas=len(batch))

        if i == 3:  # um reenvio corrompido do sistema de origem chega no 4º mês
            bad = corrupt_batch(batch.drop(columns=[TARGET]), raw)
            name = f"batch_{period}{CORRUPTED_SUFFIX}"
            bad.to_csv(PRODUCTION_DIR / f"{name}.csv", index=False)
            log_event(logger, "lote_corrompido_simulado", "warning", lote=name, linhas=len(bad))

    (PRODUCTION_DIR / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
