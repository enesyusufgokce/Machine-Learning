import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd

data = pd.read_csv("likes_training_data_realistic.csv")
model = LinearRegression()

x = data[["views", "video_age(years)", "video_duration(minutes)"]]
y = data["likes"]

polynomial_funct = PolynomialFeatures(degree=3)
polynomial_x = polynomial_funct.fit_transform(x)      # features are extended to third degree polynomial terms

scaler = StandardScaler()
scaled_x = scaler.fit_transform(polynomial_x)    # data were normalized by using z-score normalization

model.fit(scaled_x,y)

prediction_data = np.array([
                           [30639, 5, 6.22]
                           ])
prediction_init = model.predict(scaler.transform(polynomial_funct.transform(prediction_data)))

prediction = int(prediction_init[0])
print("Estimated number of likes for a video viewed {} times, uploaded {} years ago and duration of {} minutes is:".format(prediction_data[0][0],prediction_data[0][1],prediction_data[0][2]), prediction)
