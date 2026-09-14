
import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# 1. PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ==========================================
# 2. HISTORY PATH
# ==========================================

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "prediction_history.csv"
)

# ==========================================
# MODEL RESULTS PATH
# ==========================================

MODEL_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "model_comparison.csv"
)

os.makedirs(
    os.path.dirname(HISTORY_PATH),
    exist_ok=True
)


# ==========================================
# 3. MODEL PATH
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_student_model.pkl"
)


# ==========================================
# 4. LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# 5. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("🎓 Student Performance")

st.sidebar.write(
    "Student Performance Prediction System"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Dashboard",
        "🔮 Predict Performance",
        "📚 Prediction History",
        "ℹ️ About Project"
    ]
)


st.sidebar.divider()

st.sidebar.info(
    "Machine Learning based "
    "Student Performance Prediction System"
)


# ==========================================
# 7. DASHBOARD PAGE
# ==========================================

if page == "🏠 Dashboard":

    st.title(
        "🎓 Student Performance Prediction System"
    )

    st.write(
        "Welcome to the Student Performance "
        "Prediction Dashboard."
    )
        # ======================================
    # MACHINE LEARNING MODEL PERFORMANCE
    # ======================================

    st.divider()

    st.subheader(
        "🤖 Machine Learning Model Performance"
    )

    if os.path.exists(MODEL_RESULTS_PATH):

        model_results = pd.read_csv(
            MODEL_RESULTS_PATH
        )

        best_index = model_results[
            "R2 Score"
        ].idxmax()

        best_model = model_results.loc[
            best_index,
            "Model"
        ]

        best_mae = model_results.loc[
            best_index,
            "MAE"
        ]

        best_rmse = model_results.loc[
            best_index,
            "RMSE"
        ]

        best_r2 = model_results.loc[
            best_index,
            "R2 Score"
        ]

        st.success(
            f"🏆 Best Model: {best_model}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", best_mae)

        with col2:
            st.metric("RMSE", best_rmse)

        with col3:
            st.metric("R² Score", best_r2)

        st.write("### 📊 Model Comparison")

        st.dataframe(
            model_results,
            use_container_width=True
        )

        chart = model_results[
            ["Model", "R2 Score"]
        ]

        st.bar_chart(
            chart.set_index("Model")
        )

    else:

        st.warning(
            "Model performance report not found."
        )
    st.divider()

    # --------------------------------------
    # Dashboard Statistics
    # --------------------------------------

    if os.path.exists(HISTORY_PATH):

        history = pd.read_csv(
            HISTORY_PATH
        )

        total_students = len(history)

        average_score = round(
            history["Predicted Score"].mean(),
            2
        )

        highest_score = round(
            history["Predicted Score"].max(),
            2
        )

    else:

        total_students = 0
        average_score = 0
        highest_score = 0


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "👨‍🎓 Total Predictions",
            total_students
        )


    with col2:

        st.metric(
            "📊 Average Score",
            average_score
        )


    with col3:

        st.metric(
            "🏆 Highest Score",
            highest_score
        )


    st.divider()


    # --------------------------------------
    # Dashboard Information
    # --------------------------------------

    st.subheader(
        "📋 Project Overview"
    )

    st.write(
        """
        This application uses Machine Learning
        to predict a student's final score based
        on study hours, attendance, previous score,
        and sleep hours.
        """
    )


    st.subheader(
        "🔧 Input Parameters"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            "📚 Study Hours\n\n"
            "Daily study time of the student."
        )

        st.info(
            "📝 Attendance\n\n"
            "Student's attendance percentage."
        )


    with col2:

        st.info(
            "📊 Previous Score\n\n"
            "Student's previous academic score."
        )

        st.info(
            "😴 Sleep Hours\n\n"
            "Average daily sleeping hours."
        )


    st.divider()


    st.success(
        "👉 Use the sidebar to predict "
        "student performance."
    )


# ==========================================
# 8. PREDICTION PAGE
# ==========================================

elif page == "🔮 Predict Performance":

    st.title(
        "🔮 Predict Student Performance"
    )

    st.write(
        "Enter student information to predict "
        "the final score."
    )

    st.divider()


    # --------------------------------------
    # Student Name
    # --------------------------------------

    student_name = st.text_input(
        "👤 Student Name",
        placeholder="Enter student name"
    )


    # --------------------------------------
    # Student Information
    # --------------------------------------

    st.subheader(
        "📋 Student Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        study_hours = st.number_input(
            "📚 Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )


        attendance = st.number_input(
            "📝 Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )


    with col2:

        previous_score = st.number_input(
            "📊 Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )


        sleep_hours = st.number_input(
            "😴 Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )


    st.divider()


    # --------------------------------------
    # Prediction Button
    # --------------------------------------

    if st.button(
        "🔮 Predict Performance",
        use_container_width=True
    ):

        # ----------------------------------
        # Check Student Name
        # ----------------------------------

        if student_name.strip() == "":

            st.warning(
                "Please enter student name."
            )

        else:

            # --------------------------------
            # Create Student Data
            # --------------------------------

            student = pd.DataFrame({

                "study_hours": [
                    study_hours
                ],

                "attendance": [
                    attendance
                ],

                "previous_score": [
                    previous_score
                ],

                "sleep_hours": [
                    sleep_hours
                ]

            })


            # --------------------------------
            # Make Prediction
            # --------------------------------

            prediction = model.predict(
                student
            )

            score = prediction[0]


            # --------------------------------
            # Keep Score 0-100
            # --------------------------------

            score = max(
                0,
                min(100, score)
            )


            # --------------------------------
            # Grade
            # --------------------------------

            if score >= 90:

                grade = "A+"
                performance = "Excellent"

                message = (
                    "Outstanding performance! "
                    "Keep it up. 🏆"
                )


            elif score >= 80:

                grade = "A"
                performance = "Very Good"

                message = (
                    "Great performance! "
                    "Keep working consistently. 🎯"
                )


            elif score >= 70:

                grade = "B"
                performance = "Good"

                message = (
                    "Good performance. "
                    "You can improve further. 👍"
                )


            elif score >= 60:

                grade = "C"
                performance = "Average"

                message = (
                    "Try to increase study time "
                    "and attendance. 📚"
                )


            elif score >= 50:

                grade = "D"
                performance = "Below Average"

                message = (
                    "More practice and regular "
                    "study are recommended. ⚠️"
                )


            else:

                grade = "F"
                performance = "Poor"

                message = (
                    "You need significant improvement "
                    "in your study routine. 📖"
                )


            # --------------------------------
            # Prediction Result
            # --------------------------------

            st.success(
                "✅ Prediction Completed!"
            )

            st.header(
                f"🎓 Result for {student_name}"
            )


            # --------------------------------
            # Result Metrics
            # --------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Final Score",
                    f"{score:.2f}"
                )


            with col2:

                st.metric(
                    "Grade",
                    grade
                )


            with col3:

                st.metric(
                    "Performance",
                    performance
                )


            st.divider()


            # --------------------------------
            # Performance Level
            # --------------------------------

            st.subheader(
                "📈 Performance Level"
            )

            st.progress(
                int(score)
            )


            # --------------------------------
            # Student Summary
            # --------------------------------

            st.subheader(
                "📋 Student Summary"
            )


            summary = pd.DataFrame({

                "Parameter": [

                    "Study Hours",

                    "Attendance",

                    "Previous Score",

                    "Sleep Hours"

                ],

                "Value": [

                    f"{study_hours} hours",

                    f"{attendance}%",

                    f"{previous_score}",

                    f"{sleep_hours} hours"

                ]

            })


            st.table(
                summary
            )


            # --------------------------------
            # Performance Analysis
            # --------------------------------

            st.subheader(
                "📊 Performance Analysis"
            )


            chart_data = pd.DataFrame({

                "Parameter": [

                    "Study Hours",

                    "Attendance",

                    "Previous Score",

                    "Sleep Hours",

                    "Predicted Score"

                ],

                "Value": [

                    study_hours,

                    attendance,

                    previous_score,

                    sleep_hours,

                    score

                ]

            })


            st.bar_chart(
                chart_data.set_index(
                    "Parameter"
                )
            )


            # --------------------------------
            # Performance Indicators
            # --------------------------------

            st.subheader(
                "🎯 Student Performance Indicators"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    "📚 Study Hours"
                )

                st.progress(
                    min(
                        int(
                            (study_hours / 24) * 100
                        ),
                        100
                    )
                )


                st.write(
                    "📝 Attendance"
                )

                st.progress(
                    int(attendance)
                )


            with col2:

                st.write(
                    "📊 Previous Score"
                )

                st.progress(
                    int(previous_score)
                )


                st.write(
                    "😴 Sleep Hours"
                )

                st.progress(
                    min(
                        int(
                            (sleep_hours / 8) * 100
                        ),
                        100
                    )
                )


            # --------------------------------
            # Recommendation
            # --------------------------------

            st.subheader(
                "💡 Recommendation"
            )

            st.info(
                message
            )


            # --------------------------------
            # Improvement Suggestions
            # --------------------------------

            if study_hours < 4:

                st.warning(
                    "📚 Increase your daily "
                    "study hours."
                )


            if attendance < 75:

                st.warning(
                    "📝 Try to improve your "
                    "attendance."
                )


            if previous_score < 60:

                st.warning(
                    "📖 Focus on improving your "
                    "previous academic performance."
                )


            if sleep_hours < 6:

                st.warning(
                    "😴 Maintain a healthy "
                    "sleep schedule."
                )


            if (
                study_hours >= 4
                and attendance >= 75
                and previous_score >= 60
                and sleep_hours >= 6
            ):

                st.success(
                    "✅ Your study habits look good. "
                    "Maintain consistency!"
                )


            # --------------------------------
            # Save Prediction History
            # --------------------------------

            new_record = pd.DataFrame({

                "Student Name": [
                    student_name
                ],

                "Study Hours": [
                    study_hours
                ],

                "Attendance": [
                    attendance
                ],

                "Previous Score": [
                    previous_score
                ],

                "Sleep Hours": [
                    sleep_hours
                ],

                "Predicted Score": [
                    round(score, 2)
                ],

                "Grade": [
                    grade
                ],

                "Performance": [
                    performance
                ]

            })


            if os.path.exists(
                HISTORY_PATH
            ):

                history = pd.read_csv(
                    HISTORY_PATH
                )

                history = pd.concat(
                    [
                        history,
                        new_record
                    ],
                    ignore_index=True
                )

            else:

                history = new_record


            history.to_csv(
                HISTORY_PATH,
                index=False
            )


            st.success(
                "✅ Prediction saved to history!"
            )


# ==========================================
# 9. PREDICTION HISTORY PAGE
# ==========================================
# ==========================================
# 9. PREDICTION HISTORY PAGE
# ==========================================

elif page == "📚 Prediction History":

    st.title(
        "📚 Prediction History"
    )

    st.write(
        "View, download and manage all "
        "previously generated predictions."
    )

    st.divider()

    if os.path.exists(HISTORY_PATH):

        history = pd.read_csv(
            HISTORY_PATH
        )

        # ----------------------------------
        # Statistics
        # ----------------------------------

        st.subheader(
            "📊 History Statistics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "👨‍🎓 Total Students",
                len(history)
            )

        with col2:

            st.metric(
                "📈 Average Score",
                round(
                    history[
                        "Predicted Score"
                    ].mean(),
                    2
                )
            )

        with col3:

            st.metric(
                "🏆 Highest Score",
                round(
                    history[
                        "Predicted Score"
                    ].max(),
                    2
                )
            )

        st.divider()

        # ----------------------------------
        # History Table
        # ----------------------------------

        st.subheader(
            "📋 All Predictions"
        )

        st.dataframe(
            history,
            use_container_width=True
        )

        st.divider()

        # ----------------------------------
        # Download CSV
        # ----------------------------------

        st.subheader(
            "📥 Download Report"
        )

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv_data,
            file_name="student_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        # ----------------------------------
        # Student Score Chart
        # ----------------------------------

        st.subheader(
            "📈 Student Score Chart"
        )

        score_chart = history[
            [
                "Student Name",
                "Predicted Score"
            ]
        ]

        st.bar_chart(
            score_chart.set_index(
                "Student Name"
            )
        )

        st.divider()

        # ----------------------------------
        # Clear History
        # ----------------------------------

        st.subheader(
            "🗑️ Manage History"
        )

        st.warning(
            "⚠️ Clearing history will permanently "
            "delete all saved prediction records."
        )

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            os.remove(
                HISTORY_PATH
            )

            st.success(
                "✅ Prediction history cleared successfully."
            )

            st.rerun()

    else:

        st.info(
            "No prediction history available yet."
        )

# ==========================================
# 10. ABOUT PROJECT PAGE
# ==========================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About Project"
    )

    st.write(
        """
        ## 🎓 Student Performance Prediction System

        This is a Machine Learning based web
        application designed to predict student
        final performance.

        The system uses the following inputs:

        - 📚 Study Hours
        - 📝 Attendance
        - 📊 Previous Score
        - 😴 Sleep Hours

        Based on these inputs, the Machine Learning
        model predicts the student's final score.
        """
    )


    st.divider()


    st.subheader(
        "🤖 Machine Learning Model"
    )

    st.write(
        """
        The project compares different Machine
        Learning algorithms and selects the best
        performing model for prediction.

        The current project uses the selected
        best model saved as:

        best_student_model.pkl
        """
    )


    st.subheader(
        "📂 Project Components"
    )

    st.write(
        """
        • Dataset: student_data.csv

        • Model: best_student_model.pkl

        • Prediction History:
          prediction_history.csv

        • Web Application: Streamlit

        • Programming Language: Python
        """
    )


    st.divider()


    st.success(
        "🚀 Student Performance Prediction System"
        " is ready!"
    )

