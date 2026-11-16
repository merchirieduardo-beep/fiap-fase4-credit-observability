"""Privacy by Design: pseudonimização e minimização antes de qualquer uso analítico.

- customer_id  -> customer_hash (HMAC-SHA256 com salt secreto, fora do repositório)
- birth_date   -> age (generalização; a data exata é descartada)
- qualquer outra coluna de PII bruta é removida
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

import pandas as pd

from .logging_utils import get_logger, log_event

PII_COLUMNS = ["customer_id", "birth_date"]
_SALT_ENV = "PII_HASH_SALT"
_salt_cache: bytes | None = None
logger = get_logger("privacy")


def _salt() -> bytes:
    """Salt vem de variável de ambiente (nunca hardcoded nem logado)."""
    global _salt_cache
    if _salt_cache is None:
        value = os.environ.get(_SALT_ENV)
        if not value:
            value = secrets.token_hex(32)
            log_event(logger, "salt_efemero_gerado", "warning",
                      detalhe=f"{_SALT_ENV} não definido; hashes não serão estáveis entre execuções")
        _salt_cache = value.encode()
    return _salt_cache


def pseudonymize_id(value: str) -> str:
    return hmac.new(_salt(), str(value).encode(), hashlib.sha256).hexdigest()[:32]


def apply_privacy(raw: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Converte o dado bruto (com PII) no dado analítico pseudonimizado."""
    df = raw.copy()
    df.insert(0, "customer_hash", df["customer_id"].map(pseudonymize_id))
    snapshot = pd.Timestamp(snapshot_date)
    birth = pd.to_datetime(df["birth_date"])
    df["age"] = ((snapshot - birth).dt.days // 365.25).astype(int)
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
    log_event(logger, "pseudonimizacao_aplicada", linhas=len(df), colunas_removidas=PII_COLUMNS)
    return df
