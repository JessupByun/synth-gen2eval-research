import pandas as pd
import numpy as np

# Load the dataset
input_file = "data/synthetic_data/private_combined_df_cty_week2/iclr_tinypaper_tabpfn_combined_df.csv"  # Replace with actual filename
df = pd.read_csv(input_file)

# Ensure correct column order by moving 'mask_user_pct' between 'week' and 'mask_mandate'
column_order = [
    "state", "county_fips", "week", "mask_user_pct", "mask_mandate",
    "gop_vote_share_2016", "deaths_per_10k", "COVID_news", "retail_visit_per_hundred",
    "COVID_news_cable", "urban_population_percentage", "image_users", "mask_users",
    "population_density", "week_counter", "week_counter_log", "population"
]

# Convert column names if they are slightly different
df = df.rename(columns={"mask_user_pct": "mask_user_pct"})

# Fill missing values and ensure correct data types
df["county_fips"] = df["county_fips"].fillna(0).astype(int)  # Convert to integer
df["deaths_per_10k"] = df["deaths_per_10k"].fillna(0.0).astype(float)  # Default 0.0
df["COVID_news"] = df["COVID_news"].fillna(0.909090909090909).astype(float)
df["retail_visit_per_hundred"] = df["retail_visit_per_hundred"].fillna(0.909090909090909).astype(float)
df["COVID_news_cable"] = df["COVID_news_cable"].fillna(0.733788395904437).astype(float)
df["image_users"] = df["image_users"].fillna(0).astype(int)
df["mask_users"] = df["mask_users"].fillna(0).astype(int)
df["week_counter"] = df["week_counter"].fillna(1).astype(int)
df["week_counter_log"] = df["week_counter_log"].fillna(0).astype(int)
df["population"] = df["population"].fillna(0).astype(int)  # Convert to integer

# Reorder the columns
df = df[column_order]

# Save the cleaned data
output_file = "data/synthetic_data/private_combined_df_cty_week2/iclr_tinypaper_tabpfn_combined_df_formatted.csv"
df.to_csv(output_file, index=False)

print(f"Processed dataset saved as {output_file}")
