import pandas as pd
import numpy as np

np.random.seed(42)

records = 500

study_hours = np.random.uniform(1, 12, records).round(1)
attendance = np.random.uniform(50, 100, records).round(1)
previous_score = np.random.uniform(40, 95, records).round(1)
sleep_hours = np.random.uniform(5, 9, records).round(1)

final_score = (
    2.5 * study_hours
    + 0.25 * attendance
    + 0.45 * previous_score
    + 1.5 * sleep_hours
    - 20
    + np.random.normal(0, 4, records)
)

final_score = np.clip(
    final_score,
    0,
    100
).round(2)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "final_score": final_score
})

output_path = "dataset/student_data.csv"

df.to_csv(
    output_path,
    index=False
)

print("Dataset generated successfully!")
print("Total records:", len(df))
print("\nFirst 5 records:")
print(df.head())
print("\nSaved to:", output_path)