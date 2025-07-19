from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

datas = pd.read_csv("likes_training_data.csv")
model = LinearRegression()

x = datas[["views", "video_age"]]
y = datas[["likes"]]

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)
model.fit(scaled_x,y)
prediction_init = model.predict(scaler.transform([
                                                 [33690,5]
                                                 ]))
prediction = int(prediction_init[0][0])
print("Estimated number of likes for a video viewed 33690 times and uploaded 5 yeard ago is:",prediction)