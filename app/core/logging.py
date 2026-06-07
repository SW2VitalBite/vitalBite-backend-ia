"""Logging estructurado en formato JSON.

Cada registro se emite como una línea JSON con timestamp ISO-8601, nivel,
logger y mensaje, lo que facilita su ingestión por agregadores de logs en la
nube (Cloud Logging, CloudWatch, etc.). Llamar a :func:`configure_logging`
una vez durante el arranque de la aplicación.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Adjuntar atributos extra pasados con logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


_RESERVED_ATTRS = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime"}


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
