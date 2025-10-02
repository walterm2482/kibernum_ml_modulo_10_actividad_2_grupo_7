"""
train_model.py
Entrena un modelo de Regresión Logística sobre el dataset Breast Cancer (scikit-learn),
evalúa el desempeño y guarda el modelo entrenado en disco.

Ejemplo de uso
--------------
python train_model.py --log-level INFO --model-path modelo.pkl --test-size 0.2 --seed 42
"""

# ── Imports y configuración ──
import argparse
import sys
import time
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from logger_utils import setup_logger


# ── Funciones auxiliares ──
def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Objeto con los argumentos:
        - log_level : {"DEBUG","INFO","WARNING","ERROR","CRITICAL"}
        - model_path : str, ruta para guardar el modelo
        - test_size : float, proporción de test
        - seed : int, semilla para la división
    """
    p = argparse.ArgumentParser(description="Entrenamiento con logging.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    p.add_argument("--model-path", default="modelo.pkl")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def timed_step(logger, name):
    """
    Context manager para medir tiempo de ejecución y registrar logs.

    Parameters
    ----------
    logger : logging.Logger
        Instancia de logger configurado.
    name : str
        Nombre del paso a medir.

    Returns
    -------
    context manager
        Ejecuta logging de inicio, finalización y duración.
    """
    class _Ctx:
        def __enter__(self):
            self.t0 = time.perf_counter()
            logger.info(f"[{name}] inicio")
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            if exc:
                logger.error(f"[{name}] error en {dt:.3f}s", exc_info=True)
            else:
                logger.info(f"[{name}] ok en {dt:.3f}s")
    return _Ctx()


# ── Ejecución principal ──
def main():
    """
    Ejecuta el pipeline de entrenamiento y evaluación.

    Pasos
    -----
    1. Carga el dataset Breast Cancer.
    2. Divide en entrenamiento y prueba.
    3. Entrena un pipeline con escalado y Regresión Logística.
    4. Evalúa el modelo con accuracy y classification_report.
    5. Guarda el modelo en disco con joblib.
    """
    args = parse_args()
    logger = setup_logger(name="ml.train", level=args.log_level, log_path="logs/train.log")

    try:
        with timed_step(logger, "cargar_datos"):
            data = load_breast_cancer()
            X, y = data.data, data.target
            logger.info(f"n_samples={X.shape[0]} n_features={X.shape[1]} classes={set(y)}")

        with timed_step(logger, "split"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.seed, stratify=y
            )
            logger.info(f"train={X_train.shape[0]} test={X_test.shape[0]}")

        with timed_step(logger, "entrenar"):
            pipe = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
            ])
            pipe.fit(X_train, y_train)

        with timed_step(logger, "evaluar"):
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"accuracy={acc:.4f}")
            rep = classification_report(y_test, y_pred, target_names=data.target_names, digits=4)
            for line in rep.splitlines():
                logger.debug(f"report | {line.strip()}")

        with timed_step(logger, "guardar_modelo"):
            joblib.dump(pipe, args.model_path)
            logger.info(f"modelo_path='{args.model_path}'")

        logger.info("pipeline=ok")
    except Exception:
        logger.critical("pipeline=failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
