#FINAL PROJECT - FORECASTING OF THE MAXIMUM TEMPERATURE IN PEREIRA,RISARALDA,COLOMBIA.
# Authors: Isaac Pachón and Jense David Martinez.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import FFT

# Cargar datos desde el archivo CSV
df = pd.read_csv('data.csv')
df.index = pd.to_datetime(df['Fecha'])


# Crear una serie temporal a partir de los datos
series = TimeSeries.from_dataframe(df, value_cols=['Valor'], freq='D')

# Aplicar la transformada rápida de Fourier (FFT)
fft = FFT()
fft.fit(series)

# Obtener los coeficientes en el dominio de la frecuencia
trend_coefficients = fft.trend_coefficients

#Calcular las frecuencias asociadas
if trend_coefficients is None:
  print("Error: No se encontraron coeficientes de tendencia significativos.")
else:
  frequencies = np.fft.fftfreq(len(trend_coefficients), d=series.freq.delta)
  positive_frequencies = frequencies[frequencies > 0]
  print("Frecuencias dominantes: ",positive_frequencies)

# Pronóstico usando el modelo FFT
forecast = fft.predict(365*3) # Pronóstico para el próximo año (365 días)

# Visualizar el pronóstico
series.plot(label='Datos Observados')
forecast.plot(label='Pronóstico FFT')
plt.legend()
plt.show()


