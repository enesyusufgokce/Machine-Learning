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

data = pd.read_csv("animal_species.csv")

x = data[["Weight(kg)","Height(cm)","Lifespan(years)","Heart_Rate(bpm)","Body_Fat_Percent(%)","Leg_Length(cm)","Birth_Weight(kg)","Sleep_Duration(hours)"]]
y = data["Species"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

encoder = LabelEncoder()  # LabelEncoder stringleri alfabetik sıraya göre indexler
y_encoded = encoder.fit_transform(y)

model = Sequential([
    Dense(units = 4, activation = "relu"),
    Dense(units = 10, activation = "linear")
])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam())
# Adam(learning_rate=0.001) dersek yine aynı. çünkü keras, learning rate için varsayılan olarak zaten bu değeri kullanıyoo

model.fit(x_scaled, y_encoded, epochs=200)

test_data = np.array([
                     [0.62,18.0,7.5,304,10.8,11.6,0.008,12.2]
                     ])

test_data_scaled = scaler.transform(test_data)

logit = model(test_data_scaled)
probability = tf.nn.softmax(logit)
print(probability)

prediction_prob_index = np.argmax(probability)   # prediction list deki en büyük değeri tutan indexi verir
prediction = encoder.classes_[prediction_prob_index]   # LabelEncoder'ın alfabetik sıraladığı stringlerden yukraıda bulduğu indexteki stringi verir
print(prediction)
