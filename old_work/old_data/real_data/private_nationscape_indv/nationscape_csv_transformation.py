import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_label_dataset(input_csv, output_csv, test_size=0.8, random_state=42):
    # Load the entire dataset
    df = pd.read_csv(input_csv)
    
    # Split the data into train and test sets.
    # Here, "test" will contain 80% of the rows and "train" 20%.
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Add the SOURCE_LABEL column to each split.
    train_df['SOURCE_LABEL'] = "train"
    test_df['SOURCE_LABEL'] = "test"
    
    # Optionally, if you want to preserve the order from the split (i.e. first the training rows then the testing rows),
    # you can simply concatenate them. Otherwise, if you want to re-sort by the original index, you could do that.
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save the combined DataFrame to CSV.
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined dataset with SOURCE_LABEL saved to {output_csv}")

if __name__ == "__main__":
    # Change these file paths to match your environment.
    input_csv = "data/real_data/private_nationscape_indv/private_nationscape_indv_df.csv"  # This should be your whole dataset CSV.
    output_csv = "data/real_data/private_nationscape_indv/nationscape_split.csv"
    
    split_and_label_dataset(input_csv, output_csv)
