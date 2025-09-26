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
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# prepare the data
data = pd.read_csv("CoffeeRoasting.csv")

x = data[["Temperature(°C)", "Duration(minutes)"]]
y = data["Target"]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)
del x_temp, y_temp

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_cv_scaled = scaler.transform(x_cv)
x_test_scaled = scaler.transform(x_test)

# create the model
model = Sequential([
    Dense(units = 15, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(units = 15, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(units = 15, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(units = 1, activation="linear", kernel_regularizer=l2(0.001))
])

# loss and cost function
model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
# ham puan veririz, loss, sigmoidi içeride uygular [eğitimde bu.
# tahmin kısmında olasılık vermesi için 36.satırdaki ayarlamayı yaptık. çünkü output layerde linear activation var]
# (from logits demeyip output avtivation'u logistic yaparsak, loss fonksiyonuna doğrudan öncesinde hesaplanan olasılık gider, ham sayısal değer değil)
# sayısal yuvarlamayı önleyip daha doğru numeric değer için output layer activation'ı sigmoidden linear'a çevirdik, compile kısmında

# training the model
model.fit(x_train_scaled, y_train, epochs=180, validation_data=(x_cv_scaled, y_cv))       # epochs: number of steps

model.evaluate(x_test_scaled, y_test)

roasting_data = np.array([
                     [843, 12],
                     [128, 5],
                     [290, 2],
                     [321, 16],
                     [31, 4],
                     ])

roasting_data_scaled = scaler.transform(roasting_data)


# prediction
logit = model(roasting_data_scaled)   # yeni veriye göre modelin tahminini alır. ama modelin output layerinde linear activation function'ı
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