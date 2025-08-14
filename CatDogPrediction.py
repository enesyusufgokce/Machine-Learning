"""
datasets
normalization
fit
prediction
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

data = pd.read_csv("cat_dog_dataset.csv")
model = LogisticRegression()

x = data[["Weight_kg", "Lifespan_years", "HeartRate_bpm", "Gestation_days"]]
y = data["Species"]

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

model.fit(scaled_x,y)

prediction_data = np.array([
                           [33.3, 11.6, 77, 65]
                           ])

prediction_init = model.predict(scaler.transform(prediction_data))
prediction = prediction_init[0]   # use 1D array for logistic regression

print("An animal which has features as Weight:{} kg, Lifespan:{} years, HeartRate:{} bpm, "
      "Gestation:{} days is a".format(prediction_data[0][0],prediction_data[0][1],
                                 prediction_data[0][2], prediction_data[0][2]), prediction)

