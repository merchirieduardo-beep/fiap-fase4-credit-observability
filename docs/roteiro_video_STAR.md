# Roteiro do vídeo (≤ 5 min) — método STAR

Dica: deixe `reports/dashboard.html`, o relatório do Evidently e o terminal já abertos.

## S — Situation (0:00–0:40)
- Fintech de crédito com modelo de Credit Scoring em produção.
- Inflação, desemprego e novos perfis de cliente mudaram o cenário; a diretoria suspeita de
  degradação silenciosa: o modelo continua respondendo, sem erro técnico, mas aprova quem não devia.
- Risco real: a inadimplência só aparece meses depois, quando o prejuízo já está na carteira.

## T — Task (0:40–1:10)
- Interceptar dados ruins antes do modelo (contrato de dados).
- Detectar data drift e concept drift com testes estatísticos.
- Dar visibilidade operacional (logs, métricas, alertas).
- Garantir compliance com a LGPD e explicar a causa do problema.

## A — Action (1:10–3:00)
1. **Contrato (Pandera)** — mostrar `src/contracts.py` (regras R1–R7) e rodar
   `python -m src.demo_contract_failure`: lote limpo passa, lote com erros é bloqueado e vai
   para quarentena. Abrir `reports/contracts/contrato_demo_lote_com_erros.html`.
2. **Simulação de drift** — `src/simulate_drift.py`: 6 lotes mensais, intensidade de 0 a 1;
   renda nominal inflacionada, desemprego de 6% para 20%, perfil mais jovem (data drift) e
   mudança na relação X→y (concept drift).
3. **Detecção** — PSI + KS/Qui-quadrado próprios para alertas e Evidently AI para o relatório.
4. **Observabilidade** — logs JSON, histórico de métricas, alertas, MLflow e dashboard.
5. **Governança** — pseudonimização, minimização, base legal, retenção e auditoria de viés (README).

## R — Result (3:00–5:00)
- Abrir o **dashboard**: veredito "modelo degradado", AUC caindo abaixo do limite, linhas de
  risco real × previsto se afastando, inadimplência entre aprovados dobrando.
- Mostrar a **tabela de PSI** ficando vermelha mês a mês e o lote corrompido bloqueado na
  tabela de saúde do pipeline.
- Abrir o **Evidently** (`reports/evidently/`) apontando as features com drift.
- **Análise causal**: data drift sozinho não derruba o AUC; o que derruba é o concept drift.
  Logo, monitorar só a distribuição das features não basta — é preciso acompanhar calibração
  e performance com rótulos maduros.
- **Lições aprendidas**: (1) contrato na entrada evita que erro de dado vire "drift falso";
  (2) PSI avisa cedo, mas não diz se o modelo piorou; (3) rótulo atrasado exige indicadores
  antecedentes; (4) retreino precisa de gatilho objetivo e runbook.
