# ── Imagen base ──
# Usamos Python 3.11 en versión ligera (slim) para reducir tamaño
FROM python:3.11-slim

# ── Variables de entorno ──
# PYTHONDONTWRITEBYTECODE=1 → evita crear archivos .pyc
# PYTHONUNBUFFERED=1 → salida de logs sin buffer (útil en contenedores)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Dependencias del sistema ──
# Instalamos libgomp1 porque scikit-learn necesita OpenMP para cálculos paralelos
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ──
# Todo el código y archivos se copiarán a /app dentro del contenedor
WORKDIR /app

# ── Instalación de dependencias de Python ──
# Copiamos primero requirements.txt para aprovechar la cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copiar código fuente y modelo entrenado ──
COPY app.py logger_utils.py modelo.pkl ./

# ── Puerto expuesto ──
# Flask/Gunicorn usará el puerto 5000
EXPOSE 5000

# ── Comando de ejecución ──
# Usamos gunicorn (servidor de producción) para correr la API Flask
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
