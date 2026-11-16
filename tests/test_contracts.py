import pandas as pd
import pytest

from src.config import TARGET
from src.contracts import DataContractViolation, validate_batch
from src.datagen import generate_raw
from src.privacy import apply_privacy
from src.simulate_drift import corrupt_batch

SNAPSHOT = "2026-04-30"


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    raw = generate_raw(500, seed=7, snapshot_date=SNAPSHOT)
    return apply_privacy(raw, SNAPSHOT).drop(columns=[TARGET])


def test_clean_batch_passes(clean):
    assert len(validate_batch(clean, "teste_limpo")) == len(clean)


def test_corrupted_batch_is_blocked(clean):
    with pytest.raises(DataContractViolation) as exc:
        validate_batch(corrupt_batch(clean), "teste_corrompido")
    assert len(exc.value.failure_cases) > 0


@pytest.mark.parametrize("column,value", [
    ("age", 15),                      # menor de idade
    ("monthly_income", None),         # renda nula
    ("monthly_income", -10.0),        # renda negativa
    ("credit_utilization", 1.5),      # fora da faixa
    ("employment_status", "outro"),   # categoria fora do domínio
])
def test_each_rule_blocks(clean, column, value):
    bad = clean.copy()
    bad.loc[bad.index[0], column] = value
    with pytest.raises(DataContractViolation):
        validate_batch(bad, f"teste_{column}")


def test_duplicates_are_blocked(clean):
    bad = pd.concat([clean, clean.head(3)], ignore_index=True)
    with pytest.raises(DataContractViolation):
        validate_batch(bad, "teste_duplicatas")


def test_raw_pii_column_is_blocked(clean):
    bad = clean.assign(customer_id="CUST-00000001")
    with pytest.raises(DataContractViolation):
        validate_batch(bad, "teste_pii")
