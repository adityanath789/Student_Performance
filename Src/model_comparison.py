import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================
# 2. DATASET PATH
# ==========================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "student_data.csv"
)


# ==========================================
# 3. LOAD DATA
# ==========================================

df = pd.read_csv(
    DATA_PATH
)


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "sleep_hours"
    ]
]

y = df["final_score"]


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 6. MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

results = []


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({

        "Model": name,

        "MAE": round(
            mae,
            4
        ),

        "RMSE": round(
            rmse,
            4
        ),

        "R2 Score": round(
            r2,
            4
        )

    })


# ==========================================
# 8. RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(
    results
)


print("\nMODEL COMPARISON")
print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 9. FIND BEST MODEL
# ==========================================

best_index = results_df[
    "R2 Score"
].idxmax()


best_model_name = results_df.loc[
    best_index,
    "Model"
]


print("\nBEST MODEL")
print(
    "Best Model:",
    best_model_name
)


# ==========================================
# 10. TRAIN BEST MODEL
# ==========================================

best_model = models[
    best_model_name
]


best_model.fit(
    X,
    y
)


# ==========================================
# 11. SAVE BEST MODEL
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_student_model.pkl"
)


os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


joblib.dump(
    best_model,
    MODEL_PATH
)


# ==========================================
# 12. SAVE MODEL RESULTS
# ==========================================

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)


os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


MODEL_RESULTS_PATH = os.path.join(
    REPORTS_DIR,
    "model_comparison.csv"
)


results_df.to_csv(
    MODEL_RESULTS_PATH,
    index=False
)


print(
    "\nModel comparison saved to:"
)

print(
    MODEL_RESULTS_PATH
)


print(
    "\nBest model saved to:"
)

print(
    MODEL_PATH
)