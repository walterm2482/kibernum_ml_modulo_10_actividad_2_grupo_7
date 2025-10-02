# Actividad Módulo 10 - Sesión 2  
**Contenerización de una API ML con Docker**

Este proyecto entrena un modelo de **Regresión Logística** sobre el dataset *Breast Cancer* de `scikit-learn`, lo expone mediante una **API Flask**, y lo contenedoriza en **Docker**.  

---

## 📂 Archivos incluidos
- `train_model.py` → Script de entrenamiento que genera `modelo.pkl`.  
- `app.py` → API Flask con endpoints de prueba y predicción.  
- `logger_utils.py` → Configuración de logging para entrenamiento y API.  
- `requirements.txt` → Dependencias de Python.  
- `Dockerfile` → Imagen de Docker lista para ejecutar la API.  
- `modelo.pkl` → Modelo entrenado.  

---

## ⚙️ Requisitos
- Python 3.11+ (para entrenamiento local)  
- Docker (para construir y ejecutar la API en contenedor)  

---

## 🚀 Entrenamiento local
1. Crear y activar entorno virtual (opcional):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   .\venv\Scripts\activate    # Windows
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Entrenar modelo y generar `modelo.pkl`:
   ```bash
   python train_model.py
   ```

---

## 🐳 Construir imagen Docker
```bash
docker build -t ml-api .
```

---

## ▶️ Ejecutar contenedor
```bash
docker run --rm -p 5000:5000 -e LOG_LEVEL=INFO ml-api
```

---

## 🔎 Probar API
### Endpoint raíz (bienvenida y metadatos)
```bash
curl http://127.0.0.1:5000/
```

### Predicción con un vector de 30 valores
```bash
curl -s -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/json"   -d '{"input":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

#### Respuesta esperada:
```json
{
  "classes": ["malignant", "benign"],
  "predictions": [1],
  "probabilities": [[..., ...]]
}
```

---

## 📋 Notas
- No incluyas `venv/` en el repositorio (`.gitignore` recomendado).  
- `gunicorn` se usa como servidor de producción dentro del contenedor.  
- Logs se almacenan en la carpeta `logs/` (puedes mapearla con `-v` en `docker run`).  
