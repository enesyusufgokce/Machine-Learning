from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes_dataset.csv")

categorical_features = data.select_dtypes(include=["object"]).columns   # sadece OneHotEncoder'a hangi sütunları encode edeceğimizi söylüyoruz

encoder = OneHotEncoder(drop = "first", sparse_output=False)
encoded_categorical = encoder.fit_transform(data[categorical_features])  # belirttiğimiz sütunların verisini alıp encode ettik

encoded_columns = encoder.get_feature_names_out(categorical_features)
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoded_columns) # OneHotEncoder den çıkan datayı tekrar df haline getirip
# columns lara, kendi encode edilen değerlerine karşılık gelen isimlerini veriyoz

# remove categorical columns from the original data and replace them with encoded ones
data_encoded = pd.concat([data.drop(categorical_features, axis=1).reset_index(drop=True),
                          encoded_categorical_df.reset_index(drop=True)], axis=1)

x = data_encoded.drop("diagnosed_diabetes", axis=1)
y = data_encoded["diagnosed_diabetes"]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)
del x_temp, y_temp

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.4, colsample_bytree=0.6, random_state=1)
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_cv, y_cv)])

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)
print("model accuracy is:",accuracy)