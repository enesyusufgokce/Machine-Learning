import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd

data = pd.read_csv("likes_training_data_realistic.csv")
model = Ridge(alpha=1000000)

x = data[["views", "video_age(years)", "video_duration(minutes)"]]
y = data["likes"]

polynomial_funct = PolynomialFeatures(degree=27)
polynomial_x = polynomial_funct.fit_transform(x)      # features are extended to third degree polynomial terms

scaler = StandardScaler()
scaled_x = scaler.fit_transform(polynomial_x)    # data were normalized by using z-score normalization

model.fit(scaled_x,y)

prediction_data = np.array([
                            [16522,9,16.34],
                            [19704,12,7.04],
                            [11341,17,5.33],
                            [10745,15,16.54],
                            [35864,17,1.13],
                            [49699,18,8.93],
                            [21095,6,3.28],
                            [44444,22,19.92]
                           ])
prediction_init = model.predict(scaler.transform(polynomial_funct.transform(prediction_data)))

prediction1 = int(prediction_init[0])
prediction2 = int(prediction_init[1])
prediction3 = int(prediction_init[2])
prediction4 = int(prediction_init[3])
prediction5 = int(prediction_init[4])
prediction6 = int(prediction_init[5])
prediction7 = int(prediction_init[6])
prediction8 = int(prediction_init[7])

print(prediction1)
print(prediction2)
print(prediction3)
print(prediction4)
print(prediction5)
print(prediction6)
print(prediction7)
print(prediction8)