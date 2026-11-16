# Análise causal da degradação (gerado por `python -m src.causal_analysis`)

## Decomposição: data drift vs. concept drift

| Cenário | AUC | Inadimplência real | Score médio previsto | Aprovação | Inadimplência entre aprovados |
|---|---:|---:|---:|---:|---:|
| A. Baseline (sem drift) | 0.737 | 18.2% | 18.6% | 69.9% | 11.1% |
| B. Só data drift — P(X) muda | 0.769 | 24.0% | 23.4% | 60.4% | 11.7% |
| C. Só concept drift — P(y|X) muda | 0.689 | 24.2% | 18.6% | 69.9% | 17.3% |
| D. Data + concept drift | 0.701 | 29.4% | 23.4% | 60.4% | 19.1% |

## Ablação: efeito de cada mecanismo isolado (intervenção única)

| Mecanismo | ΔAUC | Δ inadimplência entre aprovados | Risco subestimado (real − previsto) |
|---|---:|---:|---:|
| Conceito: histórico (DTI/atrasos) perdeu poder preditivo | -0.038 | -5.0% | -9.1% |
| Data: maior uso do limite de crédito | -0.008 | +1.2% | -0.4% |
| Conceito: renda real corroída pela inflação | -0.007 | +6.2% | +7.5% |
| Conceito: autônomos ficaram mais arriscados | -0.006 | +4.9% | +6.0% |
| Data: mais atrasos recentes | -0.003 | +1.0% | +0.3% |
| Data: novo perfil (clientes mais jovens) | +0.000 | +0.4% | -0.5% |
| Conceito: uso do limite pesa mais no risco | +0.003 | +2.6% | +3.4% |
| Data: renda nominal inflacionada | +0.005 | -1.9% | +0.1% |
| Data: desemprego 6% → 20% | +0.033 | +0.4% | -0.4% |
