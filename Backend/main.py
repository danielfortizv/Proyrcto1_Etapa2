import re
import joblib  
import logging
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
import unicodedata
import inflect
import nltk
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from sklearn.metrics import classification_report
from datetime import datetime
from pathlib import Path
import json
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier  # Ajusta según tu modelo

# Configuración inicial
nltk.download("punkt")
nltk.download("stopwords")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de la aplicación
app = FastAPI(
    title="Fake News Detector API",
    description="API para detección y clasificación de noticias falsas en contenido político",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (Permitir solicitudes desde cualquier origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (ajusta en producción)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de rutas y archivos
MODEL_PATH = Path("modelo_lightgbm.pkl")  
MODEL_VERSION = "1.0.0"
DATA_DIR = Path("data")  
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------
# **Modelos Pydantic (Esquemas de datos)**
# ----------------------------

class NewsItem(BaseModel):
    text: str
    source: Optional[str] = None
    date: Optional[str] = None

class PredictionRequest(BaseModel):
    news_items: List[NewsItem]

class PredictionResult(BaseModel):
    prediction: int  # 0: Real, 1: Falsa
    probability: float  # Probabilidad de ser falsa
    explanation: Optional[str] = None

class PredictionResponse(BaseModel):
    results: List[PredictionResult]
    model_version: str
    timestamp: str

class TrainingDataItem(BaseModel):
    text: str
    label: int  # 0: Real, 1: Falsa
    source: Optional[str] = None
    date: Optional[str] = None

class TrainingRequest(BaseModel):
    data: List[TrainingDataItem]
    strategy: str = "incremental"  # Opciones: "incremental", "full", "partial"

class TrainingResponse(BaseModel):
    status: str
    metrics: dict  # Reporte de clasificación
    model_version: str
    new_samples: int
    strategy_used: str

# ----------------------------
# **Preprocesamiento de Texto**
# ----------------------------

p = inflect.engine()
stop_words = set(stopwords.words('spanish'))

def text_processing_pipeline(text: str) -> str:
    """Limpieza y normalización de texto"""
    if not isinstance(text, str):
        return ""
    
    # Normalización (elimina acentos y caracteres especiales)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()  # Convertir a minúsculas
    
    # Tokenización (dividir en palabras)
    tokens = word_tokenize(text)
    
    # Limpieza
    tokens = [p.number_to_words(t) if t.isdigit() else t for t in tokens]  # Convertir números a palabras
    tokens = [re.sub(r'[^\w\s]', '', t) for t in tokens]  # Eliminar puntuación
    tokens = [t for t in tokens if t and t not in stop_words]  # Eliminar stopwords
    
    return ' '.join(tokens)  # Unir tokens en un solo string

# ----------------------------
# **Carga del Modelo (Pipeline)**
# ----------------------------

def load_model():
    """Carga el pipeline completo (vectorizador + modelo)"""
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f" Pipeline cargado correctamente desde {MODEL_PATH}")
        return pipeline
    except Exception as e:
        logger.error(f" Error al cargar el modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="No se pudo cargar el modelo. Verifica que el archivo existe."
        )

# Cargar el pipeline al iniciar la API
model = load_model()

# ----------------------------
# **Endpoints de la API**
# ----------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint para predecir si una noticia es falsa o verdadera.
    """
    try:
        # 1. Preprocesar los textos
        processed_texts = [text_processing_pipeline(item.text) for item in request.news_items]
        
        # 2. Predecir usando el pipeline (automáticamente aplica TF-IDF)
        predictions = model.predict(processed_texts)
        probabilities = model.predict_proba(processed_texts)
        
        # 3. Formatear la respuesta
        results = []
        for pred, prob in zip(predictions, probabilities):
            explanation = "Noticia probablemente falsa" if pred == 1 else "Noticia probablemente verdadera"
            results.append(
                PredictionResult(
                    prediction=int(pred),
                    probability=float(prob[1]),  # Probabilidad de ser falsa
                    explanation=explanation
                )
            )
        
        return PredictionResponse(
            results=results,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f" Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la solicitud: {str(e)}"
        )

@app.post("/retrain", response_model=TrainingResponse)
async def retrain(request: TrainingRequest):
    """
    Endpoint para reentrenar el modelo con nuevos datos.
    """
    try:
        # 1. Validar la estrategia
        valid_strategies = ["incremental", "full", "partial"]
        if request.strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f" Estrategia inválida. Usa una de: {', '.join(valid_strategies)}"
            )
        
        # 2. Preprocesar los datos
        texts = [text_processing_pipeline(item.text) for item in request.data]
        labels = [item.label for item in request.data]
        
        # 3. Guardar los datos para auditoría
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_hash = hashlib.md5(json.dumps(request.dict()).encode()).hexdigest()
        data_file = DATA_DIR / f"training_data_{timestamp}_{data_hash[:8]}.json"
        
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(request.dict(), f, ensure_ascii=False)
        
        # 4. Reentrenamiento según la estrategia
        if request.strategy == "full":
            # Entrenar desde cero
            model.fit(texts, labels)
        elif request.strategy == "incremental":
            # LightGBM soporta partial_fit
            if hasattr(model.named_steps['clf'], 'partial_fit'):
                X_transformed = model.named_steps['tfidf'].transform(texts)
                model.named_steps['clf'].partial_fit(X_transformed, labels, classes=[0, 1])
            else:
                model.fit(texts, labels)
        elif request.strategy == "partial":

            model.fit(texts, labels)
        
        # 5. Guardar el modelo actualizado
        joblib.dump(model, MODEL_PATH)
        
        # 6. Evaluar el rendimiento
        preds = model.predict(texts)
        report = classification_report(labels, preds, output_dict=True)
        
        return TrainingResponse(
            status="success",
            metrics=report,
            model_version=MODEL_VERSION,
            new_samples=len(request.data),
            strategy_used=request.strategy
        )
    
    except Exception as e:
        logger.error(f" Error en reentrenamiento: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el reentrenamiento: {str(e)}"
        )

@app.get("/model_info")
async def model_info():
    """Obtener información del modelo"""
    return {
        "version": MODEL_VERSION,
        "type": str(type(model.named_steps['clf'])),  
        "features": model.named_steps['tfidf'].get_feature_names_out().tolist(), 
        "last_retrain": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat()
    }

# ----------------------------
# **Iniciar la API**
# ----------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,     
        reload=True       # Recarga automática en desarrollo
    )