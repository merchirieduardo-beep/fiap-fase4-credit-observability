"""Auditoria de viés: taxa de aprovação por grupo protegido (regra dos 4/5)."""
from __future__ import annotations

import pandas as pd

AGE_BINS = [17, 29, 44, 59, 120]
AGE_LABELS = ["18-29", "30-44", "45-59", "60+"]


def approval_by_group(scored: pd.DataFrame) -> dict:
    """`scored` precisa ter `prediction` (1 = negado), `gender` e `age`."""
    df = scored.assign(approved=(scored["prediction"] == 0).astype(float),
                       age_band=pd.cut(scored["age"], AGE_BINS, labels=AGE_LABELS))
    result = {}
    for col in ("gender", "age_band"):
        rates = df.groupby(col, observed=True)["approved"].mean()
        result[col] = {
            "approval_rate": {str(k): round(float(v), 4) for k, v in rates.items()},
            # razão entre a menor e a maior taxa de aprovação (1.0 = paridade)
            "disparate_impact": round(float(rates.min() / rates.max()), 4) if rates.max() > 0 else 0.0,
        }
    return result
