"""Etapa 1 (demonstração) — força um lote com erros e mostra o contrato BLOQUEANDO a ingestão.

Uso: python -m src.demo_contract_failure
Sai com código 1 de propósito quando o lote é bloqueado (comportamento esperado em CI/CD
de dados). Use --no-fail para sair com 0.
"""
from __future__ import annotations

import sys

import pandas as pd

from .config import REFERENCE_DIR, TARGET
from .contracts import DataContractViolation, quarantine, validate_batch
from .simulate_drift import corrupt_batch


def main(fail_exit: bool = True) -> int:
    clean = pd.read_csv(REFERENCE_DIR / "reference.csv").drop(columns=[TARGET]).head(2000)

    print("\n[1/2] Lote LIMPO passando pelo contrato...")
    validate_batch(clean, "demo_lote_limpo")
    print("      -> APROVADO: lote liberado para o modelo.")

    print("\n[2/2] Lote COM ERROS passando pelo contrato...")
    bad = corrupt_batch(clean)
    try:
        validate_batch(bad, "demo_lote_com_erros")
    except DataContractViolation as violation:
        report = quarantine(bad, violation)
        print(f"      -> BLOQUEADO: {violation}")
        print(violation.summary.to_string(index=False))
        print(f"\n      Relatório: {report}")
        print("      O lote foi para data/quarantine/ e NÃO chegou ao modelo.")
        return 1 if fail_exit else 0
    print("      -> ERRO: o lote ruim passou! O contrato precisa ser revisto.")
    return 2


if __name__ == "__main__":
    sys.exit(main(fail_exit="--no-fail" not in sys.argv))
