"""Etapa 1 — Contrato de dados (Pandera).

O contrato é a porta de entrada do pipeline: um lote que viole QUALQUER regra é
rejeitado por inteiro, movido para quarentena e nunca chega ao modelo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # pandera >= 0.24
    import pandera.pandas as pa
except ImportError:  # pandera < 0.24
    import pandera as pa

from .config import (EMPLOYMENT_STATUS, HOUSING_TYPES, LOAN_TERMS, QUARANTINE_DIR,
                     REPORTS_DIR, TARGET)
from .logging_utils import get_logger, log_event

logger = get_logger("contracts")
CONTRACT_VERSION = "1.0.0"


class DataContractViolation(Exception):
    """Lote reprovado no contrato de dados — a ingestão deve ser bloqueada."""

    def __init__(self, batch_name: str, failure_cases: pd.DataFrame):
        self.batch_name = batch_name
        self.failure_cases = failure_cases
        self.summary = (
            failure_cases.groupby(["column", "check"], dropna=False).size()
            .reset_index(name="ocorrencias").sort_values("ocorrencias", ascending=False)
        )
        super().__init__(
            f"Lote '{batch_name}' violou o contrato de dados: "
            f"{len(failure_cases)} falhas em {len(self.summary)} regras."
        )


def build_schema(with_target: bool) -> "pa.DataFrameSchema":
    """Regras rígidas do contrato.

    `strict=True` rejeita colunas fora do contrato — isso impede, por exemplo, que
    PII bruta (customer_id, birth_date) vaze para a camada analítica.
    """
    columns = {
        # R1 — identificador pseudonimizado, obrigatório e SEM DUPLICATAS
        "customer_hash": pa.Column(str, pa.Check.str_length(32, 32), nullable=False, unique=True),
        # R2 — idade: maioridade legal e limite de plausibilidade
        "age": pa.Column(int, pa.Check.in_range(18, 100), nullable=False),
        # R3 — renda: obrigatória, positiva e com teto de plausibilidade
        "monthly_income": pa.Column(float, pa.Check.in_range(1, 200_000), nullable=False),
        # R4 — demais faixas numéricas
        "employment_years": pa.Column(float, pa.Check.in_range(0, 70), nullable=False),
        "debt_to_income": pa.Column(float, pa.Check.in_range(0, 2), nullable=False),
        "credit_utilization": pa.Column(float, pa.Check.in_range(0, 1), nullable=False),
        "num_late_payments_12m": pa.Column(int, pa.Check.in_range(0, 60), nullable=False),
        "num_open_credit_lines": pa.Column(int, pa.Check.in_range(0, 80), nullable=False),
        "loan_amount": pa.Column(float, pa.Check.in_range(100, 1_000_000), nullable=False),
        # R5 — domínios categóricos fechados
        "loan_term_months": pa.Column(int, pa.Check.isin(LOAN_TERMS), nullable=False),
        "employment_status": pa.Column(str, pa.Check.isin(EMPLOYMENT_STATUS), nullable=False),
        "housing_type": pa.Column(str, pa.Check.isin(HOUSING_TYPES), nullable=False),
        "gender": pa.Column(str, pa.Check.isin(["F", "M"]), nullable=False),
    }
    if with_target:
        columns[TARGET] = pa.Column(int, pa.Check.isin([0, 1]), nullable=False)

    return pa.DataFrameSchema(
        columns=columns,
        checks=[
            # R6 — regra de negócio entre colunas: tempo de emprego compatível com a idade
            pa.Check(lambda df: df["employment_years"] <= (df["age"] - 14),
                     error="tempo_de_emprego_incompativel_com_idade"),
        ],
        strict=True,   # R7 — nenhuma coluna fora do contrato (barreira contra PII bruta)
        coerce=True,
        name=f"credit_contract_v{CONTRACT_VERSION}",
    )


def validate_batch(df: pd.DataFrame, batch_name: str, with_target: bool = False) -> pd.DataFrame:
    """Valida o lote. Retorna o DataFrame validado ou levanta DataContractViolation."""
    schema = build_schema(with_target)
    try:
        validated = schema.validate(df, lazy=True)  # lazy: coleta TODAS as falhas do lote
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases.copy()
        failures["column"] = failures["column"].fillna("<dataframe>")
        violation = DataContractViolation(batch_name, failures)
        log_event(logger, "contrato_reprovado", "error", lote=batch_name,
                  falhas=len(failures), regras_violadas=len(violation.summary),
                  contrato=CONTRACT_VERSION)
        raise violation from exc
    log_event(logger, "contrato_aprovado", lote=batch_name, linhas=len(validated),
              contrato=CONTRACT_VERSION)
    return validated


def quarantine(df: pd.DataFrame, violation: DataContractViolation) -> Path:
    """Move o lote reprovado para a quarentena e gera o relatório de violações."""
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = REPORTS_DIR / "contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = violation.batch_name
    df.to_csv(QUARANTINE_DIR / f"{name}.csv", index=False)
    violation.summary.to_csv(out_dir / f"violacoes_{name}.csv", index=False)

    rows = "".join(
        f"<tr><td>{r.column}</td><td><code>{r.check}</code></td><td class='n'>{r.ocorrencias}</td></tr>"
        for r in violation.summary.itertuples()
    )
    html = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<title>Contrato de dados — {name}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 16px;color:#1c2321}}
h1{{font-size:22px}} .bad{{color:#b3261e;font-weight:700}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}} .n{{text-align:right;font-variant-numeric:tabular-nums}}</style>
<h1>Contrato de dados <code>v{CONTRACT_VERSION}</code> — lote <code>{name}</code></h1>
<p>Status: <span class="bad">REPROVADO — ingestão bloqueada, lote em quarentena</span></p>
<p>{len(df)} linhas recebidas · {len(violation.failure_cases)} falhas · {len(violation.summary)} regras violadas</p>
<table><tr><th>Coluna</th><th>Regra violada</th><th class="n">Ocorrências</th></tr>{rows}</table></html>"""
    report = out_dir / f"contrato_{name}.html"
    report.write_text(html, encoding="utf-8")
    log_event(logger, "lote_em_quarentena", "warning", lote=name, relatorio=str(report.name))
    return report
