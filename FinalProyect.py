from matplotlib import pyplot as plt
import pandas as pd

from darts import TimeSeries
from darts.models import FFT
from darts.metrics import mae

#Importing data
# Importando datos del archivo CSV 'data.csv'
df = pd.read_csv('data.csv')
# Agrupando los datos por la columna 'Date' y sumando los valores correspondientes
df = df.groupby('Date').sum()
# Estableciendo la columna 'Date' como el índice del DataFrame
df['Date'] = df.index
# Seleccionando las columnas 'Date' y 'Value' para el DataFrame
df = df[['Date','Value']]
# Convirtiendo la columna 'Date' a formato de fecha y hora utilizando el formato '%Y-%m-%d'
df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d')

# Suavización exponencial de la columna 'Value'
df['Value'] = df['Value'].ewm(alpha=0.1 ,adjust=False).mean()


# Creación de una serie temporal a partir del DataFrame
series = TimeSeries.from_dataframe(df,
                                   time_col = 'Date',       # Columna de tiempo  
                                   value_cols = 'Value',    # Columna de valor
                                   freq='D')                # Frecuencia diaria

# Dividiendo la serie temporal en dos partes: entrenamiento (90%) y validación (10%)
train, val = series.split_before(0.9)

# Creación de un modelo de pronóstico FFT
model = FFT()
# Ajuste del modelo FFT a la serie temporal de entrenamiento
model.fit(train)
# Predicción de los valores de la serie temporal de validación
pred_val = model.predict(len(val))

# Cálculo del error absoluto medio (MAE) entre los valores reales y los valores predichos
print("MAE:", mae(pred_val, val))

# Visualización de los valores reales, los valores predichos y la serie temporal de entrenamiento
train.plot(label="train")
val.plot(label="val")
pred_val.plot(label="prediction")

# Ajuste del modelo FFT a la serie temporal completa
model.fit(series)
# Predicción de los valores futuros de la serie temporal
pred_val = model.predict(365*10)
# Visualización de la predicción futura
pred_val.plot(label="forecast")

plt.show()