#0. IMPORTY BIBLIOTEK

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import glob
import folium
import plotly.express as px
import plotly.graph_objs as go
import os
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import *
import webbrowser
from sklearn.metrics import *
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model



#9. SIEC NEURONOWA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dane = pd.read_csv(fr"C:\Users\Enter\OneDrive\Pulpit\smogData\sensors score\sensor170_mean.csv",
                               parse_dates=True,
                               index_col='UTC time')
dane = dane.dropna(how='any')
dane.rename(columns={f'170_temperature': 'Temperatura', f'170_humidity': 'Wilgotnosc', f'170_pressure': 'Cisnienie',
                     f'170_pm1': 'Stezenie PM1', f'170_pm25': 'Stezenie PM25',f'170_pm10': 'Stezenie PM10'}, inplace=True)
dane = dane.astype(int)
dane = dane.to_numpy()
X = dane[:,[1,2,3]]
y = dane[:,4]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print(X_scale)

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X_scale, y, test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

from tensorflow.keras import activations
model = Sequential()
model.add(Dense(10000, input_shape=(3,)))
model.add(layers.Activation((activations.relu)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(Dense(5000))
model.add(layers.Activation((activations.relu)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(Dense(2500))
model.add(layers.Activation((activations.relu)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(Dense(1))

sgd = tf.keras.optimizers.SGD(learning_rate=0.001, nesterov=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
rms = tf.keras.optimizers.RMSprop(learning_rate=0.001)
ada = tf.keras.optimizers.Adadelta(
    learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")
adaG = tf.keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")

#model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=500, epochs=5000, verbose=1, validation_data=(X_test,y_test), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)] )

accuracy = model.evaluate(X_val, y_val,verbose  =0)
print(accuracy)
