import pandas as pd
import csv

def coerce_schema(input_csv: str, output_csv: str) -> None:
    """
    Post-processes the survey data by:
      - Dropping rows with missing values
      - Converting each column to the appropriate data type
      - Saving the result to a new CSV file
    """
    # Read CSV
    df = pd.read_csv(input_csv)

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Convert columns to desired data types
    df["response_id"] = df["response_id"].astype(int)
    df["start_date"] = df["start_date"].astype(str)
    df["state"] = df["state"].astype(str)
    df["congress_district"] = df["congress_district"].astype(str)
    df["county"] = df["county"].astype(float)
    df["age"] = df["age"].astype(int)
    df["gender"] = df["gender"].astype(str)
    df["weight"] = df["weight"].astype(float)
    df["extra_covid_worn_mask"] = df["extra_covid_worn_mask"].astype(str)
    df["vote_2020"] = df["vote_2020"].astype(int)
    df["pid7"] = df["pid7"].astype(str)
    df["date"] = df["date"].astype(str)

    # Convert "worn" column (True/False) to boolean
    # If your CSV stores 'True'/'False' strings, .astype(bool) will work directly.
    df["worn"] = df["worn"].astype(bool)

    # Save cleaned data
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    # Print summary
    print(f"Data successfully cleaned and saved to {output_csv}")
    print(f"Rows in final dataset: {len(df)}")
    print("Column data types:")
    print(df.dtypes)

if __name__ == "__main__":
    # Manually specify file paths here
    INPUT_CSV = "/path/to/your_input_file.csv"
    OUTPUT_CSV = "/path/to/your_output_file.csv"

    coerce_schema(INPUT_CSV, OUTPUT_CSV)
