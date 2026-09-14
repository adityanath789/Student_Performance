import joblib
import os
import pandas as pd

# -----------------------------------
# 1. Project Path
# -----------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_student_model.pkl"
)


# -----------------------------------
# 2. Load Best Model
# -----------------------------------

model = joblib.load(MODEL_PATH)

print("=" * 55)
print("       STUDENT PERFORMANCE PREDICTION")
print("=" * 55)


# -----------------------------------
# 3. Take Student Input
# -----------------------------------

study_hours = float(
    input("Enter study hours: ")
)

attendance = float(
    input("Enter attendance percentage: ")
)

previous_score = float(
    input("Enter previous score: ")
)

sleep_hours = float(
    input("Enter sleep hours: ")
)


# -----------------------------------
# 4. Create Input DataFrame
# -----------------------------------

student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "sleep_hours": [sleep_hours]
})


# -----------------------------------
# 5. Predict Final Score
# -----------------------------------

prediction = model.predict(student)

score = prediction[0]


# -----------------------------------
# 6. Grade and Performance
# -----------------------------------

if score >= 90:
    grade = "A+"
    performance = "Excellent"

elif score >= 80:
    grade = "A"
    performance = "Very Good"

elif score >= 70:
    grade = "B"
    performance = "Good"

elif score >= 60:
    grade = "C"
    performance = "Average"

elif score >= 50:
    grade = "D"
    performance = "Below Average"

else:
    grade = "F"
    performance = "Poor"


# -----------------------------------
# 7. Display Result
# -----------------------------------

print("\n")
print("=" * 55)
print("              PREDICTION RESULT")
print("=" * 55)

print("Study Hours       :", study_hours)
print("Attendance        :", attendance, "%")
print("Previous Score    :", previous_score)
print("Sleep Hours       :", sleep_hours)

print("\nPredicted Final Score:", round(score, 2))
print("Grade                :", grade)
print("Performance          :", performance)

print("=" * 55)