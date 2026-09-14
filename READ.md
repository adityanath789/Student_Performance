# Student Performance Prediction System

## 1. Project Overview

Student Performance Prediction System is a Machine Learning based application
that predicts a student's final score using academic and lifestyle-related
parameters.

The system uses the following input parameters:

- Study Hours
- Attendance
- Previous Score
- Sleep Hours

The predicted final score is generated using a trained Machine Learning model.

A Streamlit web application is used to provide an interactive user interface.

---

## 2. Objectives

The main objectives of this project are:

1. Predict student final performance using Machine Learning.
2. Analyze the relationship between student parameters and final score.
3. Compare multiple Machine Learning regression algorithms.
4. Select the best performing model.
5. Provide an easy-to-use web interface.
6. Store prediction history.
7. Provide performance recommendations to students.

---

## 3. Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Python Virtual Environment

---

## 4. Machine Learning Models

The following regression models were compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

The model with the highest R² Score is selected as the best model.

---

## 5. Dataset

The dataset contains student-related features:

| Feature | Description |
|---|---|
| study_hours | Number of hours spent studying |
| attendance | Student attendance percentage |
| previous_score | Previous academic score |
| sleep_hours | Average sleeping hours |
| final_score | Final predicted/target score |

The current project uses a synthetic dataset generated using Python.

---

## 6. Project Structure

```text
Student_Performance
│
├── dataset
│   └── student_data.csv
│
├── models
│   ├── student_model.pkl
│   └── best_student_model.pkl
│
├── notebooks
│   └── EDA.ipynb
│
├── reports
│   ├── model_comparison.csv
│   └── prediction_history.csv
│
├── src
│   ├── main.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── model_comparison.py
│
├── app.py
├── generate_dataset.py
└── README.md