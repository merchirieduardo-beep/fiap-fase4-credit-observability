"""Executa o projeto de ponta a ponta. Uso: python -m src.run_all"""
from __future__ import annotations

from .config import LOG_FILE


def main() -> None:
    LOG_FILE.write_text("", encoding="utf-8")  # logs desta execução
    from . import causal_analysis, dashboard, demo_contract_failure, monitor, simulate_drift, train

    print("\n=== Etapa 1 — Contrato de dados + modelo baseline ===")
    train.main()
    demo_contract_failure.main(fail_exit=False)
    print("\n=== Etapa 2 — Simulação de drift ===")
    simulate_drift.main()
    print("\n=== Etapas 2 e 3 — Monitoramento dos lotes de produção ===")
    monitor.main()
    print("\n=== Análise causal ===")
    causal_analysis.main()
    print("\n=== Etapa 3 — Dashboard ===")
    print("Dashboard:", dashboard.main())


if __name__ == "__main__":
    main()
