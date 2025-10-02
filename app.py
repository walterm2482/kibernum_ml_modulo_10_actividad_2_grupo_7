"""
app.py
API Flask para exponer un modelo de clasificación entrenado sobre Breast Cancer.

Endpoints
---------
GET  /
    Retorna metadatos de la API y del modelo cargado.
POST /predict
    Recibe datos en formato JSON y devuelve predicciones y probabilidades.
"""

# ── Imports ──
import os
import traceback
from typing import List, Dict, Any

import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.datasets import load_breast_cancer

from logger_utils import setup_logger


# ── Configuración ──
MODEL_PATH = os.getenv("MODEL_PATH", "modelo.pkl")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = setup_logger(name="ml.api", level=LOG_LEVEL, log_path="logs/api.log")

# Metadatos del dataset
_bc = load_breast_cancer()
FEATURE_NAMES: List[str] = list(_bc.feature_names)
N_FEATURES: int = len(FEATURE_NAMES)
TARGET_NAMES: List[str] = list(_bc.target_names)


# ── Carga del modelo ──
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"modelo_cargado='{MODEL_PATH}' n_features={N_FEATURES}")
except Exception:
    logger.critical("no_se_pudo_cargar_modelo", exc_info=True)
    raise

app = Flask(__name__)


# ── Utilidades ──
def _dict_to_vector(d: Dict[str, Any]) -> List[float]:
    """
    Convierte un diccionario de características en un vector ordenado.

    Parameters
    ----------
    d : dict
        Diccionario con claves exactamente iguales a FEATURE_NAMES.

    Returns
    -------
    list of float
        Vector con valores ordenados según FEATURE_NAMES.

    Raises
    ------
    ValueError
        Si faltan o sobran campos requeridos.
    """
    missing = [k for k in FEATURE_NAMES if k not in d]
    extra = [k for k in d.keys() if k not in FEATURE_NAMES]
    if missing:
        raise ValueError(f"faltan_campos: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if extra:
        logger.warning(f"campos_extras_ignorados={extra[:5]}")
    return [float(d[k]) for k in FEATURE_NAMES]


def _parse_payload(payload: Dict[str, Any]) -> np.ndarray:
    """
    Convierte payload JSON en matriz NumPy válida para el modelo.

    Formatos aceptados
    ------------------
    {"input": [..30..]}
        Un solo vector de longitud N_FEATURES.
    {"input": {"feature_name": valor, ...}}
        Un solo diccionario con claves = FEATURE_NAMES.
    {"inputs": [[..],[..]]}
        Lote de vectores o diccionarios.

    Parameters
    ----------
    payload : dict
        Datos de entrada recibidos en JSON.

    Returns
    -------
    np.ndarray
        Matriz X de shape (n, N_FEATURES).

    Raises
    ------
    ValueError
        Si los datos no cumplen el formato esperado.
    """
    if "inputs" in payload:
        data = payload["inputs"]
    elif "input" in payload:
        data = payload["input"]
    else:
        raise ValueError("falta_campo 'input' o 'inputs'")

    if isinstance(data, dict):
        vec = _dict_to_vector(data)
        arr = np.array([vec], dtype=float)
    elif isinstance(data, list):
        if data and not isinstance(data[0], (list, tuple, dict)):
            if len(data) != N_FEATURES:
                raise ValueError(f"longitud_incorrecta: se esperan {N_FEATURES} valores")
            arr = np.array([data], dtype=float)
        else:
            rows = []
            for row in data:
                if isinstance(row, dict):
                    rows.append(_dict_to_vector(row))
                else:
                    if len(row) != N_FEATURES:
                        raise ValueError(f"fila_con_longitud_incorrecta: se esperan {N_FEATURES}")
                    rows.append([float(x) for x in row])
            arr = np.array(rows, dtype=float)
    else:
        raise ValueError("formato_no_soportado")
    return arr


# ── Endpoints ──
@app.get("/")
def root():
    """
    Endpoint raíz de la API.

    Returns
    -------
    flask.Response
        JSON con mensaje, metadatos del modelo y ejemplos de entrada.
    """
    info = {
        "message": "API de predicción operativa",
        "model_path": MODEL_PATH,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "endpoints": {
            "GET /": "mensaje de bienvenida",
            "POST /predict": "recibe JSON y devuelve predicción"
        },
        "input_formats": [
            {"input": [f"... {N_FEATURES} valores numéricos ..."]},
            {"input": {"feature_name": "valor", "...": "..."}},
            {"inputs": [[f"... {N_FEATURES} ..."], ["..."]]}
        ],
    }
    return jsonify(info), 200


@app.post("/predict")
def predict():
    """
    Endpoint de predicción.

    Returns
    -------
    flask.Response
        JSON con predicciones, clases y probabilidades.
    """
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="JSON inválido o vacío"), 400

        X = _parse_payload(payload)
        logger.info(f"predict_request n={X.shape[0]} shape={X.shape}")

        y_hat = model.predict(X).tolist()
        probs: List[List[float]] = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X).tolist()

        result = {
            "predictions": y_hat,
            "classes": TARGET_NAMES,
            "probabilities": probs if probs else None
        }
        return jsonify(result), 200

    except ValueError as ve:
        logger.warning(f"bad_request {ve}")
        return jsonify(error=str(ve)), 400
    except Exception:
        logger.error("internal_error", exc_info=True)
        return jsonify(error="internal_error", trace=traceback.format_exc()), 500


# ── Main ──
if __name__ == "__main__":
    # Para desarrollo local. En producción se recomienda gunicorn.
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
