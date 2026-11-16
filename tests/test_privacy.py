from src.datagen import generate_raw
from src.privacy import PII_COLUMNS, apply_privacy, pseudonymize_id


def test_pii_is_removed_and_id_is_pseudonymized():
    raw = generate_raw(100, seed=3, snapshot_date="2026-04-30")
    out = apply_privacy(raw, "2026-04-30")
    assert not set(PII_COLUMNS) & set(out.columns)
    assert out["customer_hash"].str.len().eq(32).all()
    assert not out["customer_hash"].str.contains("CUST").any()
    assert out["age"].between(18, 100).all()


def test_pseudonymization_is_deterministic_within_run():
    assert pseudonymize_id("CUST-1") == pseudonymize_id("CUST-1")
    assert pseudonymize_id("CUST-1") != pseudonymize_id("CUST-2")
