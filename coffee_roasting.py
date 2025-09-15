"""
binary classification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import StandardScaler

# prepare the data
data = pd.read_csv("coffee_roasting_dataset.csv")

x = data[["Temperature(°C)", "Duration(minutes)"]]
y = data["Target"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# create the model
model = Sequential([
    Dense(units = 3, activation="relu"),
    Dense(units = 1, activation="linear")])

# loss and cost function
model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam())
# ham puan veririz, loss, sigmoidi içeride uygular [eğitimde bu.
# tahmin kısmında olasılık vermesi için 36.satırdaki ayarlamayı yaptık. çünkü output layerde linear activation var]
# (from logit's demeyip output avtivation'u logistic yaparsak, loss fonksiyonuna doğrudan öncesinde hesaplanan olasılık gider, ham sayısal değer değil)
# sayısal yuvarlamayı önleyip daha doğru numeric değer için output layer activation'ı sigmoidden linear'a çevirdik, compile kısmında

# training the model
model.fit(x_scaled, y, epochs=100)       # epochs: number of steps

test_data = np.array([
                     [843, 12],
                     [128, 5],
                     [290, 2],
                     [321, 16],
                     [31, 4],
                     ])

test_data_scaled = scaler.transform(test_data)

# prediction
logit = model(test_data_scaled)   # yeni veriye göre modelin tahminini alır. ama modelin output layerinde linear activation function'ı
# olduğu için aslında bu olasılık değil, modelin, verilen etikete ait olma konusunda ne kadar emin olduğunu gösteren bir reel sayıdır

probability = tf.nn.sigmoid(logit) # logit ifadesini, sigmoid uygulayarak olasılığa çevirdik
print(probability)
# loss=MeanSquaredError compile için regressionda bunu, binary classification'da BinaryCrossentropy kullan
# hidden layers için relu kullan. doğrusal fonksiyonları uç uca birleştirip modeli o şekilde kurar. linear olmayan modeli kurabilir.
# using linear activation function in every hidden layer is the same as using no activation function in every hidden layer

for i in range(len(probability)):
    if probability[i][0] >= 0.5:
        print("the coffee is roasted")
    else:
        print("the coffee is not roasted")
