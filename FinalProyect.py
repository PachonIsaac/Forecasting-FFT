from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mlp
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from darts import TimeSeries
from darts.models import FFT, ExponentialSmoothing
from darts.metrics import mae
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

#Importing data
df = pd.read_csv('data.csv')
df = df.groupby('Date').sum()
df['Date'] = df.index
df = df[['Date','Value']]

df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d')
df.head()

#Suaviation exponential of the column Value
df['Value'] = df['Value'].ewm(alpha=0.1 ,adjust=False).mean()


#Creating time series
#df['Date'] = df['Date'].asfreq('W')
series = TimeSeries.from_dataframe(df,
                                   time_col = 'Date',  
                                   value_cols = 'Value',
                                   fill_missing_dates=True, freq='D')

series.head()




#data visualization
train, val = series.split_before(0.9)
train.plot(label="training")
val.plot(label="validation")

#prediction using FFT
model = FFT(trend="poly")
model.fit(train)
pred_val = model.predict(len(val))
print("MAE:", mae(pred_val, val))
train.plot(label="train")
val.plot(label="val")
pred_val.plot(label="prediction")

#Real life forecasting
model.fit(series)
pred_val = model.predict(len(val))
pred_val.plot(label="forecast")

plt.show()