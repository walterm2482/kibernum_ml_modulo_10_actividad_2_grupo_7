"""
logger_utils.py
Módulo de utilidades para configurar logging en proyectos de ML.
Provee setup_logger para crear loggers con salida en consola y archivo rotativo en formato JSON.
"""

# ── Imports ──
import logging
import os
import sys
import json
from logging.handlers import RotatingFileHandler


# ── Formateador personalizado ──
class JsonFormatter(logging.Formatter):
    """
    Formatea los registros de log como JSON.

    Parameters
    ----------
    logging.Formatter : clase base
        Clase de formateo de la librería estándar de logging.

    Methods
    -------
    format(record)
        Devuelve el registro de log en formato JSON serializado.
    """
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "time": self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S"),
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ── Función principal ──
def setup_logger(name: str = "ml",
                 level: str = "INFO",
                 log_path: str = "logs/app.log",
                 max_bytes: int = 1_000_000,
                 backup_count: int = 3) -> logging.Logger:
    """
    Configura un logger con salida en consola y archivo rotativo en formato JSON.

    Parameters
    ----------
    name : str, default="ml"
        Nombre del logger.
    level : str, default="INFO"
        Nivel mínimo para la consola. Valores válidos: {"DEBUG","INFO","WARNING","ERROR","CRITICAL"}.
    log_path : str, default="logs/app.log"
        Ruta del archivo donde se almacenarán los logs.
    max_bytes : int, default=1_000_000
        Tamaño máximo del archivo en bytes antes de rotar.
    backup_count : int, default=3
        Número de archivos de backup que se conservarán.

    Returns
    -------
    logging.Logger
        Logger configurado con dos handlers:
        - Consola: formato legible con timestamp.
        - Archivo: formato JSON rotativo.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Captura todo, filtran los handlers

    # Evitar duplicación si se vuelve a invocar
    if logger.handlers:
        return logger

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Handler de consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    # Handler de archivo JSON rotativo
    fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonFormatter())

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
