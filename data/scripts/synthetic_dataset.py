import numpy as np
import pandas as pd
import os

def generate_covariance_matrix(d, g):
    """
    Generate a d-dimensional covariance matrix S, where S[i, j] = g^|i−j|.
    """
    S = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            S[i, j] = g ** abs(i - j)
    return S

def generate_synthetic_data(p, n, d, g):
    """
    Generate a synthetic dataset with binary class variable Y and feature vector X.

    Parameters:
    p - Prior probability for class Y=1 (Bernoulli distribution)
    n - Number of observations
    d - Dimension of feature vector X
    g - Parameter for covariance matrix structure

    Returns:
    A pandas DataFrame containing the generated dataset.
    """

    # Generate class labels Y from Bernoulli distribution
    Y = np.random.binomial(1, p, size=n)

    # Define mean vectors for both classes
    mean_0 = np.zeros(d)  # Mean vector for Y=0: (0, ..., 0)
    mean_1 = np.array([1 / (i + 1) for i in range(d)])  # Mean vector for Y=1: (1, 1/2, 1/3, ..., 1/d)...

    # Generate covariance matrix
    S = generate_covariance_matrix(d, g)

    # Generate feature vectors X based on class label
    X = np.zeros((n, d))
    for i in range(n):
        if Y[i] == 0:
            X[i, :] = np.random.multivariate_normal(mean_0, S)
        else:
            X[i, :] = np.random.multivariate_normal(mean_1, S)

    # Create DataFrame
    column_names = [f"X{i+1}" for i in range(d)]  # Naming columns as X1, X2, ..., Xd
    df = pd.DataFrame(X, columns=column_names)
    df['target'] = Y  # Add class label column

    return df

def save_dataset(df, p, n, d, g):
    """
    Save the generated dataset to a CSV file with parameters in the filename.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    filename = f"synthetic_dataset_p{p}_n{n}_d{d}_g{g}.csv"
    file_path = os.path.join(script_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Dataset saved successfully at: {file_path}")

def main(p, n, d, g):
    """
    Main function to generate the synthetic dataset with given parameters.

    Parameters:
    p - Prior probability for class Y=1 (0 < p < 1)
    n - Number of observations
    d - Dimension of feature vector X
    g - Parameter for covariance matrix (0 < g < 1)
    """
    if not (0 < p < 1) or not (0 < g < 1):
        raise ValueError("p and g must be between 0 and 1.")

    # Generate synthetic dataset
    df = generate_synthetic_data(p, n, d, g)

    # Save dataset with parameters in the filename
    save_dataset(df, p, n, d, g)

    print(f"Dataset generated successfully with shape: {df.shape}")
    print(df.head())

    return df  # Return the dataset if needed in further analysis

if __name__ == "__main__":
    # Example run -> replace these values or call main() with different ones
    main(0.6, 500, 20, 0.8)