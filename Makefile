PY ?= python

.PHONY: install all train contract-demo simulate monitor causal dashboard test mlflow-ui clean

install:
	$(PY) -m pip install -r requirements.txt

all:            ## pipeline completo (Etapas 1 a 3 + analise causal)
	$(PY) -m src.run_all

train:          ## Etapa 1: referencia + contrato + baseline
	$(PY) -m src.train

contract-demo:  ## Etapa 1: lote com erros sendo bloqueado (sai com codigo 1 de proposito)
	-$(PY) -m src.demo_contract_failure

simulate:       ## Etapa 2: gera os lotes de producao com drift
	$(PY) -m src.simulate_drift

monitor:        ## Etapas 2-3: contrato -> scoring -> drift -> alertas -> Evidently
	$(PY) -m src.monitor

causal:
	$(PY) -m src.causal_analysis

dashboard:      ## Etapa 3: reports/dashboard.html
	$(PY) -m src.dashboard

test:
	$(PY) -m pytest -q

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns

clean:
	rm -rf data models mlruns reports monitoring .pytest_cache
