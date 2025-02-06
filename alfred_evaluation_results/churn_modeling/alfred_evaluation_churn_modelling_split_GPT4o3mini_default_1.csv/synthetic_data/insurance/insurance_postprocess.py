import pandas as pd

def coerce_schema(input_csv: str, output_csv: str) -> None:
    """
    Reads a dataset with columns:
        "age","sex","bmi","children","smoker","region","charges"
    and converts each to the correct data type:
        - age: integer
        - sex: string
        - bmi: float (rounded to 2 decimals)
        - children: integer
        - smoker: string
        - region: string
        - charges: float (rounded to 2 decimals)

    Then saves the cleaned data to 'output_csv'.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Convert columns to desired data types
    df["age"] = df["age"].astype(int)
    df["sex"] = df["sex"].astype(str)
    df["bmi"] = df["bmi"].astype(float).round(2)
    df["children"] = df["children"].astype(int)
    df["smoker"] = df["smoker"].astype(str)
    df["region"] = df["region"].astype(str)
    df["charges"] = df["charges"].astype(float).round(2)

    # Save to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Data successfully coerced and saved to {output_csv}")

if __name__ == "__main__":
    # Specify your file paths here:
    INPUT_CSV = "data/synthetic_data/insurance/iclr_tinypaper_tabsyn_insurance.csv"
    OUTPUT_CSV = "data/synthetic_data/insurance/iclr_tinypaper_tabsyn_formatted_insurance.csv"

    # Call the function with your specified paths
    coerce_schema(INPUT_CSV, OUTPUT_CSV)
