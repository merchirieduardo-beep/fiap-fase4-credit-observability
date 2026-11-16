#!/usr/bin/env bash
# Cria o histórico de commits semânticos, em ordem lógica de construção.
# Uso: bash scripts/git_commits.sh   (na raiz do projeto, DEPOIS de rodar `make all`)
# Este script só cria commits locais; não configura remoto nem faz push.
set -euo pipefail

if [ ! -f reports/dashboard.html ]; then
  echo "Rode 'make all' antes, para gerar os relatórios que serão versionados." >&2
  exit 1
fi

[ -d .git ] || git init -b main

c() { git commit -q -m "$1" && echo "ok: $1"; }

git add .gitignore requirements.txt Makefile .env.example src/__init__.py src/config.py
c "chore: estrutura inicial, dependências fixadas e configuração central"

git add src/logging_utils.py
c "feat(logging): logs estruturados em JSON centralizados"

git add src/privacy.py
c "feat(privacy): pseudonimização com HMAC e minimização de PII"

git add src/datagen.py
c "feat(data): gerador do dataset de crédito de referência"

git add src/contracts.py
c "feat(contracts): contrato de dados com Pandera e quarentena de lotes"

git add src/model.py src/train.py src/fairness.py
c "feat(model): treino do baseline de credit scoring com model card"

git add src/simulate_drift.py src/demo_contract_failure.py
c "feat(drift): simulação de data drift e concept drift em lotes mensais"

git add src/drift.py
c "feat(drift): detecção com PSI, KS, qui-quadrado e relatórios Evidently"

git add src/observability.py src/monitor.py
c "feat(monitoring): pipeline de monitoramento, alertas, MLflow e Prometheus"

git add src/dashboard.py
c "feat(dashboard): dashboard HTML de saúde do pipeline e do modelo"

git add src/causal_analysis.py src/run_all.py
c "feat(causal): análise causal da degradação por intervenção"

git add tests
c "test: cobre contrato de dados, testes de drift e privacidade"

git add .github
c "ci: executa testes e pipeline completo a cada push"

git add README.md docs scripts
c "docs: governança, LGPD, análise causal e roteiro do vídeo STAR"

git add reports monitoring
c "chore(reports): relatórios de drift, contrato, dashboard e métricas gerados"

echo; git log --oneline
