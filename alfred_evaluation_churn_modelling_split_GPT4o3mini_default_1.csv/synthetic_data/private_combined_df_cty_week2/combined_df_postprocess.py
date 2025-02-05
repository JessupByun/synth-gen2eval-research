#!/usr/bin/env python3

import pandas as pd
import csv

def coerce_schema(input_csv: str, output_csv: str) -> None:
    """
    Reads a dataset with columns:
        "state","county_fips","week","mask_user_pct","mask_mandate",
        "gop_vote_share_2016","deaths_per_10k","COVID_news","retail_visit_per_hundred",
        "COVID_news_cable","urban_population_percentage","image_users","mask_users",
        "population_density","week_counter","week_counter_log","population"

    - Drops rows with any missing (NaN) values
    - Coerces each column to the appropriate data type
    - Removes extra quotes/spaces from string columns
    - Writes the cleaned dataset to 'output_csv'
    """

    # 1. Read CSV with skipinitialspace=True to handle spaces after commas
    df = pd.read_csv(input_csv, skipinitialspace=True, quotechar='"')

    # 2. Drop rows with any missing values
    df.dropna(inplace=True)

    # 3. Convert columns to desired data types
    df["state"] = df["state"].astype(str)
    df["county_fips"] = df["county_fips"].astype(int)
    df["week"] = df["week"].astype(str)
    df["mask_user_pct"] = df["mask_user_pct"].astype(float).round(4)
    df["mask_mandate"] = df["mask_mandate"].astype(str)
    df["gop_vote_share_2016"] = df["gop_vote_share_2016"].astype(float).round(3)
    df["deaths_per_10k"] = df["deaths_per_10k"].astype(float).round(6)
    df["COVID_news"] = df["COVID_news"].astype(float).round(6)
    df["retail_visit_per_hundred"] = df["retail_visit_per_hundred"].astype(float).round(3)
    df["COVID_news_cable"] = df["COVID_news_cable"].astype(float).round(6)
    df["urban_population_percentage"] = df["urban_population_percentage"].astype(float).round(3)
    df["image_users"] = df["image_users"].astype(int)
    df["mask_users"] = df["mask_users"].astype(int)
    df["population_density"] = df["population_density"].astype(float).round(1)
    df["week_counter"] = df["week_counter"].astype(int)
    df["week_counter_log"] = df["week_counter_log"].astype(float).round(3)
    df["population"] = df["population"].astype(int)

    # 4. Strip out any leftover extra quotes/spaces in string columns
    string_cols = ["state", "week", "mask_mandate"]
    for col in string_cols:
        df[col] = df[col].str.strip().str.replace('"', '')

    # 5. Write to CSV with minimal quoting to avoid doubling quotes
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    # Print summary
    print(f"Data successfully cleaned and saved to {output_csv}")
    print(f"Rows in final dataset: {len(df)}")
    print("Column types:")
    print(df.dtypes)

if __name__ == "__main__":
    # Manually specify file paths
    INPUT_CSV = "data/synthetic_data/private_combined_df_cty_week2/iclr_tinypaper_tabpfn_combined_df.csv"
    OUTPUT_CSV = "data/synthetic_data/private_combined_df_cty_week2/iclr_tinypaper_tabpfn_combined_df_formatted.csv"

    coerce_schema(INPUT_CSV, OUTPUT_CSV)
