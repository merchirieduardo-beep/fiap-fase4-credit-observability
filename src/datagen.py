"""Gerador do dataset de crédito (sintético, sem dados pessoais reais).

O processo gerador é conhecido, o que permite simular separadamente:
- DATA DRIFT    (muda P(X)): renda nominal inflacionada, mais desemprego, etc.
- CONCEPT DRIFT (muda P(y|X)): a mesma renda nominal compra menos, autônomos ficam
  mais arriscados, utilização de crédito pesa mais, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EMPLOYMENT_STATUS, HOUSING_TYPES, LOAN_TERMS

INFLATION = 0.60  # inflação acumulada no cenário de drift máximo

DATA_DRIFT_KEYS = ["income", "unemployment", "utilization", "age", "late_payments"]
CONCEPT_DRIFT_KEYS = ["real_income", "self_employed_risk", "utilization_weight",
                      "history_signal_decay"]


def uniform(keys: list[str], t: float) -> dict[str, float]:
    return {k: float(t) for k in keys}


def sample_features(n: int, rng: np.random.Generator, drift: dict[str, float] | None = None) -> pd.DataFrame:
    d = {**{k: 0.0 for k in DATA_DRIFT_KEYS}, **(drift or {})}

    # novo perfil de cliente: mais jovem
    age = np.clip(rng.normal(41 - 7 * d["age"], 12, n), 18, 80).astype(int)

    # desemprego sobe de 6% para 20%
    probs = np.array([0.55 - 0.14 * d["unemployment"], 0.20, 0.10, 0.09,
                      0.06 + 0.14 * d["unemployment"]])
    status = rng.choice(EMPLOYMENT_STATUS, size=n, p=probs / probs.sum())

    mult = pd.Series(status).map({"clt": 1.0, "autonomo": 0.95, "servidor_publico": 1.3,
                                  "aposentado": 0.8, "desempregado": 0.45}).to_numpy(dtype=float)
    # renda NOMINAL cresce com a inflação
    income = rng.lognormal(np.log(4200), 0.55, n) * mult * (1 + INFLATION * d["income"])
    income = np.clip(income, 600, 80_000).round(2)

    emp_years = np.minimum(rng.exponential(6, n), age - 18)
    emp_years = np.where(status == "desempregado", 0.0, emp_years).round(1)

    dti = np.clip(rng.beta(2, 5, n) * 1.2, 0, 2).round(3)
    util = np.clip(rng.beta(2 + 1.3 * d["utilization"], 4, n), 0, 1).round(3)
    late = rng.poisson(0.6 + 0.35 * d["late_payments"], n)
    lines = 1 + rng.poisson(3.5, n)
    loan = np.clip(income * rng.lognormal(np.log(4), 0.5, n), 500, 300_000).round(2)
    term = rng.choice(LOAN_TERMS, size=n, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    housing = rng.choice(HOUSING_TYPES, size=n, p=[0.45, 0.35, 0.20])
    gender = rng.choice(["F", "M"], size=n)

    return pd.DataFrame({
        "age": age, "monthly_income": income, "employment_years": emp_years,
        "debt_to_income": dti, "credit_utilization": util,
        "num_late_payments_12m": late, "num_open_credit_lines": lines,
        "loan_amount": loan, "loan_term_months": term,
        "employment_status": status, "housing_type": housing, "gender": gender,
    })


def default_probability(df: pd.DataFrame, concept: dict[str, float] | None = None) -> np.ndarray:
    """P(default | X). `concept` controla a intensidade de cada mecanismo de concept drift."""
    c = {**{k: 0.0 for k in CONCEPT_DRIFT_KEYS}, **(concept or {})}
    real_income = df["monthly_income"] / (1 + INFLATION * c["real_income"])
    unemp = (df["employment_status"] == "desempregado").astype(float)
    selfemp = (df["employment_status"] == "autonomo").astype(float)
    public = (df["employment_status"] == "servidor_publico").astype(float)
    z = (
        -3.2
        + (2.0 - 1.6 * c["history_signal_decay"]) * df["debt_to_income"]
        + (1.6 + 0.8 * c["utilization_weight"]) * df["credit_utilization"]
        + (0.45 - 0.40 * c["history_signal_decay"]) * df["num_late_payments_12m"]
        - 0.9 * np.log(real_income / 4200)
        - 0.03 * df["employment_years"]
        + 1.0 * unemp
        + (0.30 + 1.7 * c["self_employed_risk"]) * selfemp
        - 0.4 * public
        - 0.012 * (df["age"] - 40)
        + 0.25 * np.log(df["loan_amount"] / real_income / 4)
    )
    return 1 / (1 + np.exp(-z.to_numpy(dtype=float)))


def generate_raw(n: int, seed: int, snapshot_date: str, id_offset: int = 0,
                 data_drift: dict[str, float] | None = None,
                 concept_drift: dict[str, float] | None = None) -> pd.DataFrame:
    """Gera o dado BRUTO, como chegaria do sistema de origem (contém PII sintética)."""
    rng = np.random.default_rng(seed)
    df = sample_features(n, rng, data_drift)
    df["default"] = (rng.random(n) < default_probability(df, concept_drift)).astype(int)

    snapshot = pd.Timestamp(snapshot_date)
    days = (df["age"] * 365.25 + rng.integers(1, 360, n)).astype(int)
    df.insert(0, "birth_date", (snapshot - pd.to_timedelta(days, unit="D")).dt.strftime("%Y-%m-%d"))
    df.insert(0, "customer_id", [f"CUST-{id_offset + i:08d}" for i in range(n)])
    return df.drop(columns=["age"])  # idade é derivada depois, na camada de privacidade
