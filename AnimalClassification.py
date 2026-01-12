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
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping



data = pd.read_csv("animal_species.csv")

x = data[["Weight(kg)","Height(cm)","Lifespan(years)","Heart_Rate(bpm)","Body_Fat_Percent(%)","Leg_Length(cm)","Birth_Weight(kg)","Sleep_Duration(hours)"]]
y = data["Species"]

encoder = LabelEncoder()  # LabelEncoder stringleri alfabetik sıraya göre indexler
y_encoded = encoder.fit_transform(y)

# 60% train, 20% validation, 20% test split
x_train, x_temp, y_train, y_temp = train_test_split(x, y_encoded, test_size=0.4, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)
del x_temp, y_temp

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)    # fit and transform. learn the parameters according to x_train and transform it by using this learned parameters
x_val_scaled = scaler.transform(x_val)            # just use the learned parameters to transform the x_val
x_test_scaled = scaler.transform(x_test)          # same as x_val. just transform, to prevent data leaking


model = Sequential([
    Dense(units = 16, activation = "relu", kernel_regularizer =l2(0.0001)),
    Dense(units = 16, activation = "relu", kernel_regularizer =l2(0.0001)),
    Dense(units = 10, activation = "linear", kernel_regularizer=l2(0.0001))
])
# Tahmin edilecek 10 class olduğu için output layerde 10 unit bulunur. Her unit in belirli bir class ı temsil
# edebilmesi, SparseCategoricalCrossentropy fonksiyonunun, label encoding den elde edilen label ları doğrudan output
# layerdeki nöronların indeksleriyle eşleştirmesinden kaynaklanıyor


# hesaplama hızı ve vanishing gradient durumunu önlemel için hidden layerlerde sigmoid yerine relu activation function'ı kullandık

# her bir neuronun activation functionundaki weightler random initialization ile başlatıldığı için bir layerdeki aynı input vektörünü
# alan nöronların aynı sonucu üretmesi önlenip ve farklı featureler keşfetmesi sağlanıyor

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
# yukarıda label encoder kullandığımız için loss fonksiyonunu SparseCategoricalCrossentropy olarak seçtik

early_stopping = EarlyStopping(
    monitor='val_loss',      # Takip edilecek değer (validation loss)
    patience=15,             # 15 epoch boyunca hata düşmezse durdur
    mode='min',              # monitor değerinin azalmasını bekliyoruz
    restore_best_weights=True # Durduğunda, hatanın en düşük olduğu andaki ağırlıkları yükle
)

model.fit(x_train_scaled, y_train, epochs=230, validation_data=(x_val_scaled, y_val), callbacks=[early_stopping])

model.evaluate(x_test_scaled, y_test)   # see the accuracy (indicated in the compile part) and the loss of the test set

# statistical report (Precision, Recall, F1-Score, Support)
y_test_probs = model.predict(x_test_scaled)
y_test_predict_idx = np.argmax(y_test_probs, axis=1)
print(classification_report(y_test, y_test_predict_idx, target_names=encoder.classes_))

animal_data = np.array([
                       [0.54,24.3,6.4,328,6.6,12.9,0.014,12.0],
                       [10.07,38.4,3.2,102,7.5,17.5,0.141,10.7],
                       [3.5, 50, 14, 110, 12, 25, 0.1, 14],
                       [2, 40, 9, 180, 8, 25, 0.05, 12],
                       [60, 90, 15, 70, 10, 40, 3, 8],
                       ])

animal_data_scaled = scaler.transform(animal_data)

logit = model.predict(animal_data_scaled)
probability = tf.nn.softmax(logit)
print(probability)       # sum of the probabilities is 1

for i in range(len(probability)):
    prediction_prob_index = np.argmax(probability[i])   # prediction list deki en büyük değeri tutan indexi verir
    prediction = encoder.classes_[prediction_prob_index]   # LabelEncoder'ın alfabetik sıraladığı stringlerden yukarıda bulduğu indexteki stringi veriyor
    print(prediction)

#NOTE:
# Overfitting/underfitting control was performed by monitoring the loss and accuracy values in the compile stage
# Detailed performance metrics of the model (Precision, Recall, F1) were computed using the test data after training