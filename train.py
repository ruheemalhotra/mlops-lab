import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

os.makedirs("model", exist_ok=True)

data = pd.read_csv("data/data.csv")
X = data[["x"]]
y = data["y"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model/model.pkl")

print("Model trained!")
