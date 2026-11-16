"""Configuração central do projeto (caminhos, features e limiares de alerta)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"                # dado bruto COM PII (nunca versionado)
REFERENCE_DIR = DATA / "reference"    # Dataset de Referência (treino)
PRODUCTION_DIR = DATA / "production"  # Dataset de Análise/Produção (onde o drift ocorre)
QUARANTINE_DIR = DATA / "quarantine"  # lotes bloqueados pelo contrato
SCORED_DIR = DATA / "scored"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MONITORING_DIR = ROOT / "monitoring"
LOG_FILE = MONITORING_DIR / "logs" / "pipeline.jsonl"
METRICS_FILE = MONITORING_DIR / "metrics_history.jsonl"
ALERTS_FILE = MONITORING_DIR / "alerts.jsonl"

SEED = 42
N_REFERENCE = 20_000
BATCH_SIZE = 5_000
REFERENCE_SNAPSHOT = "2026-04-30"
# Lotes mensais de produção: a intensidade do drift cresce de 0 (1º mês) a 1 (último mês)
PRODUCTION_BATCHES = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]

ID_COLUMN = "customer_hash"
TARGET = "default"
NUM_FEATURES = [
    "age", "monthly_income", "employment_years", "debt_to_income",
    "credit_utilization", "num_late_payments_12m", "num_open_credit_lines",
    "loan_amount", "loan_term_months",
]
CAT_FEATURES = ["employment_status", "housing_type"]
FEATURES = NUM_FEATURES + CAT_FEATURES
# Atributo protegido: usado SOMENTE em auditoria de viés, nunca como feature do modelo
AUDIT_COLUMNS = ["gender"]

EMPLOYMENT_STATUS = ["clt", "autonomo", "servidor_publico", "aposentado", "desempregado"]
HOUSING_TYPES = ["propria", "alugada", "familiar"]
LOAN_TERMS = [12, 24, 36, 48, 60]

# Limiares de monitoramento
PSI_WARN = 0.10        # 0.10 <= PSI < 0.20 -> atenção
PSI_DRIFT = 0.20       # PSI >= 0.20 -> drift
KS_ALPHA = 0.05        # p-valor do teste KS / qui-quadrado
DRIFT_SHARE_ALERT = 0.30   # fração de features com drift que dispara recomendação de retreino
AUC_DROP_ALERT = 0.05      # queda absoluta de AUC vs. baseline
CALIBRATION_GAP_ALERT = 0.04     # inadimplência real − score médio previsto (sinal de concept drift)
DISPARATE_IMPACT_MIN = 0.80      # regra dos 4/5 na taxa de aprovação entre grupos
FAIRNESS_DEGRADATION = 0.10      # piora tolerada da razão de impacto vs. baseline

for _d in (RAW_DIR, REFERENCE_DIR, PRODUCTION_DIR, QUARANTINE_DIR, SCORED_DIR,
           MODELS_DIR, REPORTS_DIR, MONITORING_DIR, LOG_FILE.parent):
    _d.mkdir(parents=True, exist_ok=True)
