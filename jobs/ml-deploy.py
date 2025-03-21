# Pipeline de Coleta, Transformação e Modelagem Preditiva em Tempo Real com Spark Streaming e TensorFlow
# Pipeline de Deploy do Modelo

# Ajusta o nível de log do TensorFlow
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sqlite3
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, FloatType, StringType, TimestampType
from pyspark.ml.feature import VectorAssembler
from tensorflow.keras.models import load_model
import numpy as np
import json

spark = SparkSession.builder.appName("pl-deploy").getOrCreate()

schema = StructType([
    StructField("id_sensor", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("nivelco2", FloatType(), True),
    StructField("nivelpm10", FloatType(), True),
    StructField("nivelmetano", FloatType(), True)
])

# Carregando o modelo LSTM salvo para detecção de anomalias
model = load_model("/opt/spark/modelos/modelo_sil_gru.keras")
print("\nModelo GRU para detecção de anomalias carregado com sucesso.")

# Carregando os parâmetros de padronização
try:
    with open("/opt/spark/modelos/scaler_params.json", "r") as f:
        scaler_params = json.load(f)
        mean = np.array(scaler_params["mean"])
        std = np.array(scaler_params["std"])
    print("Parâmetros de padronização carregados com sucesso.\n")
except Exception as e:
    print(f"Erro ao carregar os parâmetros de padronização: {e}")
    exit(1)

# Função de detecção de anomalias com cálculo de erro de reconstrução e salvamento no banco de dados SQLite
def sil_detecta_anomalias(batch_df, batch_id):

    conn = sqlite3.connect("/opt/spark/modelos/sildb.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sil_previsoes (
            id_sensor TEXT,
            timestamp TEXT,
            error REAL,
            is_anomaly INTEGER
        )
    ''')
    
    # Converte para vetor numpy
    batch_np = np.array([row['features'].toArray() for row in batch_df.collect()])

    if batch_np.size == 0:
        print(f"Batch {batch_id} está vazio. Nada para processar.")
        conn.close()
        return

    # Caso tenha uma única amostra, transformar para 2D
    if batch_np.ndim == 1:
        batch_np = batch_np.reshape(1, -1)  

    # Aplica a padronização
    batch_np = (batch_np - mean) / std

    # Ajustar para o formato esperado pelo GRU
    batch_np = batch_np.reshape(-1, 1, 3)  

    # Realiza a previsão com o modelo carregado
    predictions = model.predict(batch_np)

    # Calcula o erro de reconstrução
    threshold = 0.8 
    for i, pred in enumerate(predictions):
        error = np.mean(np.abs(pred - batch_np[i]))
        is_anomaly = int(error > threshold)
        sensor_id = batch_df.collect()[i]['id_sensor']
        timestamp = batch_df.collect()[i]['timestamp']
        
        # Exibe anomalia detectada
        if is_anomaly:
            print(f"Anomalia detectada no batch {batch_id} para o {sensor_id} com erro {error}")
        else:
            print(f"Anomalia não detectada no batch {batch_id}.")
        
        cursor.execute('''
            INSERT INTO sil_previsoes (id_sensor, timestamp, error, is_anomaly)
            VALUES (?, ?, ?, ?)
        ''', (sensor_id, timestamp, error, is_anomaly))
        conn.commit()

    conn.close()

# Configuração do stream de dados usando socket
streaming_data = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Transformando os dados de streaming e convertendo colunas para DoubleType
sensor_data = streaming_data \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("nivelco2", col("nivelco2").cast("double")) \
    .withColumn("nivelpm10", col("nivelpm10").cast("double")) \
    .withColumn("nivelmetano", col("nivelmetano").cast("double"))

# Preparação dos dados para o modelo
assembler = VectorAssembler(inputCols = ["nivelco2", "nivelpm10", "nivelmetano"], outputCol = "features")
assembled_data = assembler.transform(sensor_data)

# Aplicação do modelo no stream
print("\nPronto Para Realizar Detecção de Anomalias em Tempo Real...\n")
query = assembled_data.writeStream \
    .foreachBatch(sil_detecta_anomalias) \
    .start()

query.awaitTermination()

spark.stop()
