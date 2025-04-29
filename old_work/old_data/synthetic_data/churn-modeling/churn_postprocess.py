import pandas as pd

def coerce_schema(input_csv: str, output_csv: str) -> None:
    """
    Reads a dataset with columns:
        "RowNumber","CustomerId","Surname","CreditScore","Geography","Gender","Age",
        "Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"

    Converts data types to:
        - RowNumber: integer
        - CustomerId: integer
        - Surname: string
        - CreditScore: integer
        - Geography: string
        - Gender: string
        - Age: integer
        - Tenure: integer
        - Balance: float (rounded to 2 decimals)
        - NumOfProducts: integer
        - HasCrCard: integer (0 or 1)
        - IsActiveMember: integer (0 or 1)
        - EstimatedSalary: float (rounded to 2 decimals)
        - Exited: integer (0 or 1)
    
    Drops rows with any missing/null values.

    Saves the cleaned dataset to 'output_csv'.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Drop rows with missing/null values
    df.dropna(inplace=True)

    # Convert columns to desired data types
    df["RowNumber"] = df["RowNumber"].astype(int)
    df["CustomerId"] = df["CustomerId"].astype(int)
    df["Surname"] = df["Surname"].astype(str)
    df["CreditScore"] = df["CreditScore"].astype(int)
    df["Geography"] = df["Geography"].astype(str)
    df["Gender"] = df["Gender"].astype(str)
    df["Age"] = df["Age"].astype(int)
    df["Tenure"] = df["Tenure"].astype(int)
    df["Balance"] = df["Balance"].astype(float).round(2)
    df["NumOfProducts"] = df["NumOfProducts"].astype(int)
    df["HasCrCard"] = df["HasCrCard"].astype(int)  # Expecting 0 or 1
    df["IsActiveMember"] = df["IsActiveMember"].astype(int)  # Expecting 0 or 1
    df["EstimatedSalary"] = df["EstimatedSalary"].astype(float).round(2)
    df["Exited"] = df["Exited"].astype(int)  # Expecting 0 or 1

    # Save to a new CSV
    df.to_csv(output_csv, index=False)

    # Print summary
    print(f"Data successfully cleaned and saved to {output_csv}")
    print(f"Total Rows Processed: {len(df)}")
    print(f"Column Data Types:\n{df.dtypes}")

if __name__ == "__main__":
    # Specify your file paths here:
    INPUT_CSV = "data/synthetic_data/churn-modeling/iclr_tinypaper_tabsyn_churn.csv"
    OUTPUT_CSV = "data/synthetic_data/churn-modeling/iclr_tinypaper_tabsyn_churn_formatted.csv"

    # Call the function with your specified paths
    coerce_schema(INPUT_CSV, OUTPUT_CSV)
