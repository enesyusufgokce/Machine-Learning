"""
multiclass classification
"""

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

data = pd.read_csv("animal_species.csv")

x = data[["Weight(kg)","Height(cm)","Lifespan(years)","Heart_Rate(bpm)","Body_Fat_Percent(%)","Leg_Length(cm)","Birth_Weight(kg)","Sleep_Duration(hours)"]]
y = data["Species"]

encoder = LabelEncoder()  # LabelEncoder stringleri alfabetik sıraya göre indexler
y_encoded = encoder.fit_transform(y)

# 60% train, 20% cross validation, 20% test split
x_train, x_temp, y_train, y_temp = train_test_split(x, y_encoded, test_size=0.4, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)
del x_temp, y_temp

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)    # fit and transform. learn the parameters according to x_train and transform it by using this learned parameters
x_cv_scaled = scaler.transform(x_cv)              # just use the learned parameters to transform the x_cv
x_test_scaled = scaler.transform(x_test)          # same as x_cv. just transform, to prevent data leaking


model = Sequential([
    Dense(units = 16, activation = "relu", kernel_regularizer =l2(0.0001)),
    Dense(units = 16, activation = "relu", kernel_regularizer =l2(0.0001)),
    Dense(units = 10, activation = "linear", kernel_regularizer=l2(0.0001))
])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
# Adam(learning_rate=0.001) dersek yine aynı. çünkü keras, learning rate için varsayılan olarak zaten bu değeri kullanıyo

model.fit(x_train_scaled, y_train, epochs=230, validation_data=(x_cv_scaled, y_cv))

model.evaluate(x_test_scaled, y_test)   # see the accuracy (indicated in the compile part) and the loss of the test set

animal_data = np.array([
                       [0.54,24.3,6.4,328,6.6,12.9,0.014,12.0],
                       [10.07,38.4,3.2,102,7.5,17.5,0.141,10.7],
                       [3.5, 50, 14, 110, 12, 25, 0.1, 14],
                       [2, 40, 9, 180, 8, 25, 0.05, 12],
                       [60, 90, 15, 70, 10, 40, 3, 8],
                       [0.5, 25, 6, 350, 5, 10, 0.02, 14],
                       [8, 60, 12, 100, 15, 30, 0.5, 9],
                       [7.67,37.7,14.9,90,17.1,23.0,0.314,14.0],
                       [14.68,52.0,13.0,107,20.7,25.5,0.537,12.9],
                       [3.61,26.0,16.0,165,12.4,14.1,0.119,11.9],
                       [3.92,27.1,14.2,125,7.3,11.3,0.106,11.6]
                       ])

animal_data_scaled = scaler.transform(animal_data)

logit = model(animal_data_scaled)
probability = tf.nn.softmax(logit)
print(probability)       # sum of the probabilities is 1

for i in range(len(probability)):
    prediction_prob_index = np.argmax(probability[i])   # prediction list deki en büyük değeri tutan indexi verir
    prediction = encoder.classes_[prediction_prob_index]   # LabelEncoder'ın alfabetik sıraladığı stringlerden yukarıda bulduğu indexteki stringi verir
    print(prediction)
