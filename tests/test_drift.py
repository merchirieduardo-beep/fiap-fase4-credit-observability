import numpy as np
import pandas as pd

from src.drift import compute_drift, psi_categorical, psi_numeric


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, b) < 0.02


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(1, 1, 5000))
    assert psi_numeric(a, b) > 0.25


def test_psi_categorical_detects_new_mix():
    a = pd.Series(["x"] * 900 + ["y"] * 100)
    b = pd.Series(["x"] * 500 + ["y"] * 500)
    assert psi_categorical(a, b) > 0.25


def test_compute_drift_flags_only_drifted_column():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 4000), "b": rng.normal(0, 1, 4000),
                        "c": rng.choice(["x", "y"], 4000)})
    cur = pd.DataFrame({"a": rng.normal(1.2, 1, 4000), "b": rng.normal(0, 1, 4000),
                        "c": rng.choice(["x", "y"], 4000)})
    table = compute_drift(ref, cur, numeric=["a", "b"], categorical=["c"]).set_index("feature")
    assert table.loc["a", "status"] == "drift"
    assert table.loc["b", "status"] == "estavel"
    assert table.loc["c", "status"] == "estavel"
