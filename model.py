# model.py
import pandas as pd
from sqlalchemy import create_engine, exc, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time

# Inicializar el modelo de RandomForest

def fetch_data():
    # Crear el motor de conexión
    """Ejecuta una consulta SQL y convierte el resultado a un DataFrame."""
    
    MYSQL_USER = "consumptionservice"
    MYSQL_PORT = 3306
    MYSQL_DB = "consumption_db"
    
    cadena = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    print(cadena)
    
    engine = create_engine(cadena)
    
    print("engine creado")
    
    query = text("SELECT user_id, device_id AS id_dispositivo, value, start_time, end_time FROM ConsumptionEntries where user_id = '20815639-1010-4ec8-bd2f-6b33aa3cd1ea';")
    
    print("Intentando conectar a la base de datos...")
    with engine.connect() as connection:
        print("Conexión establecida, ejecutando consulta SQL...")
        result = connection.execute(query)
        df_f = pd.DataFrame(result.fetchall(), columns=result.keys())

    return df_f

def preprocess_data(df_f):
    """Preprocesa el DataFrame para preparar los datos."""
    df_f['start_time'] = pd.to_datetime(df_f['start_time'])
    df_f['end_time'] = pd.to_datetime(df_f['end_time'])
    df_f['duration'] = (df_f['end_time'] - df_f['start_time']).dt.total_seconds() / 3600
    df_f['month'] = df_f['start_time'].dt.month
    df_f['day_of_week'] = df_f['start_time'].dt.day_of_week
    df_f['start_hour'] = df_f['start_time'].dt.hour
    df_f['end_hour'] = df_f['end_time'].dt.hour
    df_f = df_f.drop(columns=['end_time', 'start_time'])
    df_f['is_weekend'] = df_f['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Ajuste de horas
    df_f["end_hour"] = df_f[["start_hour", "end_hour"]].max(axis=1)
    df_f["start_hour"] = df_f[["start_hour", "end_hour"]].min(axis=1)

    # Agrupación por usuario y día
    df_daily = df_f.groupby(["user_id", "month", "day_of_week"]).agg(
        daily_consumption=("value", "sum"),
        avg_value=("value", "mean"),
        avg_duration=("duration", "mean")
    ).reset_index()
    
    return df_daily

def train_model():
    """Carga los datos, preprocesa y entrena el modelo."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    df_f = fetch_data()
    #df_f = pd.read_csv("consumptionDB.csv") #OJO: Cambiar por fetch_data()
    df_daily = preprocess_data(df_f)

    X = df_daily[["month", "day_of_week", "avg_value", "avg_duration"]]
    y = df_daily["daily_consumption"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    
    '''
    import numpy as np

    # Valores reales y predichos (supongamos que ya los tienes)
    y_true = y_test  # Valores reales
    y_pred = model.predict(X_test)  # Valores predichos por el modelo

    # Calcular el MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"MAPE: {mape:.2f}%")
    '''

    return model  # Retorna el modelo entrenado


def predict_consumption(user_id, month, day_of_week, model):
    df_f = fetch_data()
    #df_f = pd.read_csv("consumptionDB.csv") #OJO cambiar
    df_daily = preprocess_data(df_f)

    avg_value = df_daily[df_daily["user_id"] == user_id]["avg_value"].mean()
    avg_duration = df_daily[df_daily["user_id"] == user_id]["avg_duration"].mean()

    input_data = pd.DataFrame({
        "month": [month],
        "day_of_week": [day_of_week],
        "avg_value": [avg_value],
        "avg_duration": [avg_duration]
    })

    return model.predict(input_data)[0]

