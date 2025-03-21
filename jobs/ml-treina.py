# Pipeline de Coleta, Transformação e Modelagem Preditiva em Tempo Real com Spark Streaming e TensorFlow
# Pipeline de Treinamento do Modelo

# Ajusta o nível de log do TensorFlow
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType, TimestampType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, RepeatVector, TimeDistributed

spark = SparkSession.builder.appName("pipeline-treinamento").getOrCreate()

# Schema para dados de sensores
schema = StructType([
    StructField("id_sensor", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("nivelco2", FloatType(), True),
    StructField("nivelpm10", FloatType(), True),
    StructField("nivelmetano", FloatType(), True)
])

# Carregar e preparar dados
csv_file_path = "/opt/spark/dados/dataset.csv"

def sil_carrega_dados(spark, file_path, schema):
    sensor_df = spark.read.csv(file_path, schema = schema, header = True)
    print(f"\nTotal de registros carregados: {sensor_df.count()}\n")
    return sensor_df

def sil_prepara_dados(sensor_df, scaler_model = None):

    # Cria o vetor de atributos
    assembler = VectorAssembler(inputCols = ["nivelco2", "nivelpm10", "nivelmetano"], outputCol = "features")
    assembled_data = assembler.transform(sensor_df)
    
    # Padronização (somente com dados de treino)
    if scaler_model is None:

        scaler = StandardScaler(inputCol = "features", outputCol = "scaled_features", withMean = True, withStd = True)
        scaler_model = scaler.fit(assembled_data)

        mean = scaler_model.mean.toArray()
        std = scaler_model.std.toArray()
        
        # Salvar os valores de média e desvio padrão aprendidos pelo padronizador em um arquivo JSON
        with open("/opt/spark/modelos/scaler_params.json", "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
        
        print("\nParâmetros de padronização salvos com sucesso.\n")
    
    # Aplica o scaler_model para padronizar os dados
    scaled_data = scaler_model.transform(assembled_data).cache()
    
    return scaled_data, scaler_model

# Função para criar o modelo GRU
# O modelo está prevendo a reconstrução dos dados de entrada e usando o erro de reconstrução para detectar anomalias. 
# Se o erro for pequeno, os dados são considerados normais; se o erro for grande, é um sinal de que há uma anomalia nos dados. 
# Esse tipo de abordagem é muito comum em aplicações de detecção de falhas em sistemas IoT, onde dados de sensores são monitorados 
# em tempo real para identificar comportamentos incomuns.
# Estamos usando aqui aprendizado não supervisionado.
def cria_modelo_anomalia(input_shape):

    # Cria a sequência de camadas da rede neural
    model = Sequential()
    
    # Codificador
    model.add(Input(shape = input_shape))
    model.add(GRU(64, activation = 'relu', return_sequences = False))
    
    # Repetir o vetor para reconstruir
    model.add(RepeatVector(input_shape[0]))
    
    # Decodificador
    model.add(GRU(64, activation = 'relu', return_sequences = True))
    
    # Camada de saída para reconstruir cada passo de tempo e recurso
    model.add(TimeDistributed(Dense(input_shape[1])))
    
    # Compilar o modelo para reconstrução dos dados de entrada
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model

# Carrega os dados
sensor_data = sil_carrega_dados(spark, csv_file_path, schema)

# Divide os dados em treinamento (90%) e teste (10%)
train_data = sensor_data.sample(fraction = 0.9, seed = 123)
test_data = sensor_data.subtract(train_data)

# Prepara os dados de treinamento e aplica o padronizador
train_data_prepared, scaler_model = sil_prepara_dados(train_data)

# Prepara os dados de teste usando o mesmo scaler_model
test_data_prepared, _ = sil_prepara_dados(test_data, scaler_model)

# Extrai e prepara os dados para o treinamento do modelo
X_train_df = train_data_prepared.select("scaled_features").toPandas()
X_train = np.array(X_train_df["scaled_features"].tolist()).reshape(-1, 1, 3)

# Extrai e prepara os dados para avaliação do modelo
X_test_df = test_data_prepared.select("scaled_features").toPandas()
X_test = np.array(X_test_df["scaled_features"].tolist()).reshape(-1, 1, 3)

# Cria o modelo GRU 
modelo_sil_anomalia = cria_modelo_anomalia((1, 3))

# Treinamento do modelo
modelo_sil_anomalia.fit(X_train, X_train, epochs = 15, batch_size = 1, verbose = 2)

# Avaliação do modelo no conjunto de teste
print("\nAvaliação do modelo com dados de teste.\n")
loss = modelo_sil_anomalia.evaluate(X_test, X_test, verbose = 2)
print(f"\nErro no conjunto de teste: {loss}")

# Salvando o modelo
modelo_sil_anomalia.save("/opt/spark/modelos/modelo_sil_gru.keras")
print("\nModelo GRU para detecção de anomalias salvo com sucesso.\n")

# Encerrar sessão Spark
spark.stop()
