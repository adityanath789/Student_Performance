import pandas as pd
import os

# Project folder ka path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset ka path
DATA_PATH = os.path.join(BASE_DIR, "dataset", "student_data.csv")

# CSV load
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 50)

# 1. First 5 records
print("\n1. FIRST 5 RECORDS")
print(df.head())

# 2. Dataset shape
print("\n2. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Column names
print("\n3. COLUMN NAMES")
print(df.columns.tolist())

# 4. Data types
print("\n4. DATA TYPES")
print(df.dtypes)

# 5. Missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

# 6. Duplicate rows
print("\n6. DUPLICATE ROWS")
print(df.duplicated().sum())

# 7. Statistical summary
print("\n7. STATISTICAL SUMMARY")
print(df.describe())

print("\n" + "=" * 50)
print("DATASET CHECK COMPLETED")
print("=" * 50)