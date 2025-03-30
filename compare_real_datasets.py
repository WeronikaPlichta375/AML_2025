import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)

from LogRegCCD_script import LogRegCCD

def load_data(data_path):
    """Load data and split into train/test sets"""
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def sklearn_logreg_validation(X_train, y_train, X_test, y_test, metrics):
    """Sklearn logistic regression validation"""
    model = LogisticRegression(max_iter=1000, penalty=None)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    

    results = {}
    for metric in metrics:
        if metric == "roc_auc":
            score = roc_auc_score(y_test, probs)
        else:
            if metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_test, y_pred)
            elif metric == "precision":
                score = precision_score(y_test, y_pred)
            elif metric == "recall":
                score = recall_score(y_test, y_pred)
            elif metric == "f1":
                score = f1_score(y_test, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        results[metric] = score
    
    results_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    coefficients = model.coef_[0]

    return results_df, coefficients


def save_coefficients(output_dir, model_name, coefficients, feature_names=None):
    """Save model coefficients to a text file"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{model_name}_coefficients.txt")
    
    with open(filename, 'w') as f:
        f.write(f"{model_name} Coefficients\n")
        f.write("="*50 + "\n")
        f.write(f"{'Bias Term':<15}: {coefficients[0]:.6f}\n")
        for i, value in enumerate(coefficients[1:]):
            name = f"Feature {i+1}" if feature_names is None else feature_names[i]
            f.write(f"{name:<15}: {value:.6f}\n")
            
        f.write("\n")

def compare_logistic_regressions(data_path, output_dir):
    """Comparison function based on "roc_auc", "balanced_accuracy", "precision", "recall", "f1"""
    print("Starting script...")
    
    os.makedirs(output_dir, exist_ok=True)
    original_stdout = sys.stdout
    
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        sys.stdout = f
        try:
            X_train, X_test, y_train, y_test = load_data(data_path)
            print("Data loaded successfully!", file=original_stdout)
            
            try:
                feature_names = pd.read_csv(data_path).columns[:-1].tolist()
            except Exception:
                feature_names = None
            
            metrics = ["roc_auc", "balanced_accuracy", "precision", "recall", "f1"]

            print("\n====== Sklearn Logistic Regression ======")
            sk_results, sk_coef = sklearn_logreg_validation(X_train, y_train, 
                                                           X_test, y_test, metrics)
            print("\nSklearn Validation Results:")
            print(sk_results.to_string(index=False))
            save_coefficients(output_dir, "sklearn", sk_coef, feature_names)

            print("\n====== LogRegCCD Implementation ======")
            model = LogRegCCD(
                lambdas=np.logspace(-4, 0, 10),
                max_iter=1000,
                tol=1e-6
            )
            
            print("Training custom model...", file=original_stdout)
            model.fit(X_train, y_train)
            
            print("Making predictions...", file=original_stdout)
            probs = model.predict_proba(X_test, lmbd=0.1)
            print("Predictions (first 10):", probs[:10])
            
            print("Validating model...", file=original_stdout)
            validation_results = {}
            for metric in metrics:
                best_lambda, best_score = model.validate(X_test, y_test, metric=metric)
                validation_results[metric] = {
                    "Best Lambda": best_lambda,
                    "Best Score": best_score
                }
            
            print("\nCustom Model Validation Results:")
            df_results = pd.DataFrame.from_dict(validation_results, orient="index")
            print(df_results)
            
            print("Saving coefficients...", file=original_stdout)

            for lmbd, coefs in model.coefs.items():
                save_coefficients(output_dir, f"custom_lambda_{lmbd:.4f}", coefs, feature_names)
            
            for metric in metrics:
                best_lambda = validation_results[metric]["Best Lambda"]
                if best_lambda in model.coefs:
                    save_coefficients(output_dir, f"custom_best_{metric}", 
                                    model.coefs[best_lambda], feature_names)

        finally:
            sys.stdout = original_stdout
    
    print(f"\nScript completed, Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare custom and sklearn logistic regression.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    compare_logistic_regressions(args.data, args.output)