import os
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

def generate_synthetic_data(df):
    """
    Generates synthetic data using TabPFN on the provided DataFrame.
    The rightmost column is forced to be the target.
    
    Features (all columns except the rightmost):
      - Numeric features are kept continuous.
      - Categorical features are converted to integer codes.
    Target (the rightmost column):
      - If numeric, it is used as-is.
      - If non-numeric, it is converted to category codes (with a mapping for decoding).
    
    The TabPFNClassifier is trained on the processed features and target.
    Synthetic features are generated by sampling uniformly within the observed range for
    numeric features and uniformly among integer codes for categorical features.
    The synthetic target is predicted using TabPFN.
    
    Returns a synthetic DataFrame with the same number of rows as the original,
    with the target as the rightmost column.
    """
    # Force the rightmost column to be the target.
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    # Separate features and target.
    features = df[feature_cols].copy()
    target = df[target_col].copy()

    # Identify numeric and categorical features.
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

    # Convert categorical features to integer codes.
    processed_features = features.copy()
    for col in categorical_features:
        processed_features[col] = features[col].astype('category').cat.codes

    # Process target.
    if pd.api.types.is_numeric_dtype(target):
        target_processed = target.values  # Use numeric target as-is.
        target_mapping = None
    else:
        target_cat = target.astype('category')
        target_processed = target_cat.cat.codes.values  # Encode target.
        target_mapping = dict(enumerate(target_cat.cat.categories))

    # Train TabPFN.
    X_train = processed_features
    y_train = pd.Series(target_processed, index=target.index)
    model = TabPFNClassifier(device='cpu')
    model.fit(X_train, y_train)

    # Generate synthetic features.
    n_synth = df.shape[0]
    synthetic_features = pd.DataFrame(columns=X_train.columns)
    for col in X_train.columns:
        col_min = X_train[col].min()
        col_max = X_train[col].max()
        if col in numeric_features:
            # Sample continuous values uniformly.
            synthetic_features[col] = np.random.uniform(col_min, col_max, size=n_synth)
        else:
            # For categorical features, sample integer codes uniformly.
            synthetic_features[col] = np.random.randint(col_min, col_max + 1, size=n_synth)

    # Predict synthetic target.
    synthetic_y_pred = model.predict(synthetic_features)

    # Decode synthetic target if it's categorical.
    if pd.api.types.is_numeric_dtype(target):
        synthetic_target = synthetic_y_pred
    else:
        synthetic_target = [target_mapping.get(code, np.nan) for code in synthetic_y_pred]

    # Decode categorical features back to original categories.
    synthetic_features_decoded = synthetic_features.copy()
    for col in categorical_features:
        original_categories = features[col].astype('category').cat.categories
        def map_cat_code(x):
            if 0 <= x < len(original_categories):
                return original_categories[x]
            else:
                return np.nan
        synthetic_features_decoded[col] = synthetic_features_decoded[col].apply(map_cat_code)
    
    # Reassemble synthetic DataFrame.
    synthetic_df = synthetic_features_decoded.copy()
    synthetic_df[target_col] = synthetic_target
    synthetic_df = synthetic_df[list(feature_cols) + [target_col]]
    return synthetic_df

def process_csv_file(input_csv, output_csv, batch_size=200):
    """
    Loads and shuffles a CSV (using seed=42). If the DataFrame has <= batch_size rows, 
    it is processed in one go. Otherwise, it is partitioned into chunks of up to batch_size rows.
    Each chunk is processed, then the results are concatenated.
    """
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_rows = df.shape[0]

    if n_rows <= batch_size:
        synthetic_df = generate_synthetic_data(df)
    else:
        synthetic_batches = []
        for start in range(0, n_rows, batch_size):
            batch_df = df.iloc[start:start+batch_size]
            synthetic_batch = generate_synthetic_data(batch_df)
            synthetic_batches.append(synthetic_batch)
        synthetic_df = pd.concat(synthetic_batches, ignore_index=True)

    if synthetic_df.shape[0] != n_rows:
        print(f"[WARNING] Synthetic row count ({synthetic_df.shape[0]}) differs from original ({n_rows}).")

    synthetic_df.to_csv(output_csv, index=False)
    print(f"[INFO] Synthetic data saved to {output_csv}")

def process_dataset(dataset_name, generator_name="tabpfn", batch_size=200):
    """
    Processes all CSV files found in:
        LTM_data/LTM_real_data/{dataset_name}/train/
    Synthetic CSVs are saved under:
        LTM_data/LTM_synthetic_data/LTM_tabpfn_synthetic_data/synth_{dataset_name}/
    Each output file is named:
        {original_csv_filename}_{generator_name}_default_0.csv
    """
    real_data_path = os.path.join("LTM_data", "LTM_real_data", dataset_name)
    train_folder = os.path.join(real_data_path, "train")
    if not os.path.isdir(train_folder):
        print(f"[ERROR] Train folder not found: {train_folder}")
        return

    # Updated path for TabPFN synthetic data:
    synthetic_folder = os.path.join(
        "LTM_data", "LTM_synthetic_data", "LTM_tabpfn_synthetic_data", f"synth_{dataset_name}"
    )
    os.makedirs(synthetic_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(train_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"[WARNING] No CSV files found in {train_folder}.")
        return

    for csv_file in csv_files:
        input_csv_path = os.path.join(train_folder, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        output_csv_name = f"{base_name}_{generator_name}_default_0.csv"
        output_csv_path = os.path.join(synthetic_folder, output_csv_name)

        print(f"[INFO] Processing: {input_csv_path} -> {output_csv_path}")
        process_csv_file(input_csv_path, output_csv_path, batch_size=batch_size)

if __name__ == "__main__":
    dataset_name = "airfoil-self-noise"
    process_dataset(dataset_name, generator_name="tabpfn", batch_size=200)