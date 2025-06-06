import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer

def load_dataset():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset.
    Returns a DataFrame with features and target.
    """
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df

def ensure_sufficient_features(df):
    """
    Ensure that the number of features is at least 50% of the number of observations.
    If not, function adds dummy variables - permuted copies of original variables.
    It returns the modified DataFrame.
    """
    num_samples, num_features = df.shape[0], df.shape[1] - 1  # Excluding target
    required_features = num_samples // 2

    print(f"Number of features before adding dummy variables: {num_features}")

    if num_features < required_features:
        additional_features = []
        for i in range(required_features - num_features):
            # Select a random existing feature (excluding target)
            col_to_duplicate = np.random.choice(df.columns[:-1])
            # Create a permuted copy
            new_col = np.random.permutation(df[col_to_duplicate].values)
            # Store the new column
            additional_features.append(pd.Series(new_col, name=f"{col_to_duplicate}_perm_{i}"))

        # Add new dummy variables to the DataFrame
        df = pd.concat([df] + additional_features, axis=1)

    num_features_after_dummy = df.shape[1] - 1
    print(f"Number of features after adding dummy variables: {num_features_after_dummy}")

    return df

def fill_missing_values(df):
    """
    Fill missing values in the dataset with the median of each column.
    It returns the modified DataFrame.
    """
    df.fillna(df.median(), inplace=True)
    return df

def remove_collinear_features(df, threshold=0.9):
    """
    Remove highly collinear features - correlation above the given threshold.
    Returns the modified DataFrame.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

    df.drop(columns=collinear_features, inplace=True)
    df = df.copy()  # Defragment DataFrame to improve performance

    print(f"Number of features after removing collinearity: {df.shape[1] - 1}")

    return df

def save_dataset(df, filename="prepared_breast_cancer.csv"):
    """
    Save the processed dataset to a CSV file in the same directory as this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    file_path = os.path.join(script_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Dataset saved successfully at: {file_path}")

def prepare_dataset():
    """
    Load, preprocess, and return the dataset ready for logistic regression.
    Includes filling missing values, adding dummy variables (if necessary), 
    removing collinear features, and saving the dataset as a CSV file.
    """
    df = load_dataset()
    df = ensure_sufficient_features(df)
    df = fill_missing_values(df)
    df = remove_collinear_features(df)
    
    print(f"Final dataset shape: {df.shape}")
    print(df.head())
    print('Unique values ​​of the target column: ', df['target'].unique())
    save_dataset(df)

    return df

if __name__ == "__main__":
    prepared_data = prepare_dataset()