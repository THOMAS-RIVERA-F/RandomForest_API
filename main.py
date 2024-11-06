# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import predict_consumption, train_model
import datetime

from fastapi.middleware.cors import CORSMiddleware

# Configura la aplicación FastAPI como antes
app = FastAPI()

# Permitir CORS para cualquier origen (o especifica tu origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Cambia esto al puerto correcto de tu frontend
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST"],  # Específicamente incluir OPTIONS
    allow_headers=["*"],  # Permite todos los headers
)

class ConsumptionPredictionRequest(BaseModel):
    user_id: str
    date: str  # fecha en formato 'YYYY-MM-DD'

class ConsumptionPredictionResponse(BaseModel):
    user_id: str
    date: str
    predicted_consumption: float


#---------
# Agrega una variable global para el modelo
model = None

@app.on_event("startup")
async def startup_event():
    global model  # Usa global para acceder a la variable
    try:
        print("Cargando datos y entrenando el modelo en el evento de inicio...")
        model = train_model()  # Asigna el modelo a la variable global
        print("Datos cargados y modelo entrenado exitosamente.")
    except Exception as e:
        print(f"Error en el evento de inicio: {e}")

    
@app.post("/predict_consumption", response_model=ConsumptionPredictionResponse)
async def predict_consumption_endpoint(request: ConsumptionPredictionRequest):
    global model  # Accede al modelo global
    try:
        date = datetime.datetime.strptime(request.date, "%Y-%m-%d")
        month = date.month
        day_of_week = date.weekday()
        
        predicted_value = predict_consumption(request.user_id, month, day_of_week, model)
        
        return ConsumptionPredictionResponse(
            user_id=request.user_id,
            date=request.date,
            predicted_consumption=predicted_value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

