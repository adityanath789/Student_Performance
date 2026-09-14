import pandas as pd
import os


def load_data():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(
        BASE_DIR,
        "dataset",
        "student_data.csv"
    )

    df = pd.read_csv(DATA_PATH)

    return df


def clean_data(df):

    # Duplicate rows remove
    df = df.drop_duplicates()

    # Missing values handle
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    return df


if __name__ == "__main__":

    df = load_data()

    print("Original Dataset:")
    print(df.shape)

    df = clean_data(df)

    print("\nCleaned Dataset:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())