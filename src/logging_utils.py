"""Logs estruturados (JSON Lines) centralizados em monitoring/logs/pipeline.jsonl."""
import json
import logging
from datetime import datetime, timezone

from .config import LOG_FILE


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "component": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ConsoleFields(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        fields = getattr(record, "fields", {})
        record.fields_txt = " ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s %(fields_txt)s"))
    console.addFilter(_ConsoleFields())
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, level: str = "info", **fields) -> None:
    """Registra um evento com campos estruturados. Nunca passe PII ou segredos aqui."""
    getattr(logger, level)(event, extra={"fields": fields})
