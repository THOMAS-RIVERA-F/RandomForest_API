# Archivo config.py

MYSQL_USER = "consumption_service"
MYSQL_PASSWORD = "consumption_service_password"
MYSQL_HOST = "34.45.188.83"
MYSQL_PORT = "3306"
MYSQL_DB = "consumption_db"

# Construcción de la URL de conexión para SQLAlchemy
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
