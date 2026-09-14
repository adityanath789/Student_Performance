from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Dataset path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "student_data.csv"
)


# Load dataset
df = pd.read_csv(DATA_PATH)


# Features
X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "sleep_hours"
    ]
]


# Target
y = df["final_score"]


# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Total records:", len(df))

print("Training records:", len(X_train))

print("Testing records:", len(X_test))


# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

print("\nModel training completed!")

# Prediction on test data
predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions[:10])

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)


print("\nMODEL EVALUATION")
print("----------------------")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

import joblib
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_model.pkl"
)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)