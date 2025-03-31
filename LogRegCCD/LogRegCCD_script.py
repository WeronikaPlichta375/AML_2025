import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, balanced_accuracy_score

class LogRegCCD:
    def __init__(self, lambdas=None, tol=1e-6, max_iter=1000):
        """
        Initializes the Lasso Logistic Regression model.

        Parameters:
        - lambdas: list of lambda values (L1 regularization strengths)
        - tol: tolerance for convergence (based on cost change)
        - max_iter: max number of iterations for coordinate descent
        """
        self.lambdas = lambdas # List of regularization strengths
        self.tol = tol # Stopping threshold
        self.max_iter = max_iter # Max number of iterations per lambda
        self.coefs = {} # Dictionary to store learned weights
        self.training_history = {}  # for storing per-lambda training info

    def _sigmoid(self, z):
        """
        Sigmoid activation function (with protection from overflow).
        """
        z = np.clip(z, -500, 500) # Avoid overflow in exp()
        return 1 / (1 + np.exp(-z)) # Standard sigmoid

    def _loss(self, X, y, w, lmbd):
        """
        Calculates logistic loss + L1 penalty.

        Parameters:
        - X: input features (with bias)
        - y: binary labels (0 or 1)
        - w: current weights
        - lmbd: regularization strength

        Returns:
        - total cost (log loss + L1)
        """
        probs = self._sigmoid(X @ w) # Predicted probabilities
        log_loss = -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9)) # Cross-entropy
        l1_penalty = lmbd * np.sum(np.abs(w)) # L1 regularization term
        return log_loss + l1_penalty

    def fit(self, X, y):
        """
        Trains the model using coordinate descent for each lambda.

        Automatically adds bias term as first column.
        """
        X = np.c_[np.ones(X.shape[0]), X] # Add bias term (column of ones)
        y = y.reshape(-1) # Flatten y to 1D
        n, d = X.shape # n=samples, d=features (+1 for bias)

        if self.lambdas is None:
            # If no lambda list given, generate a default log-spaced list
            self.lambdas = np.logspace(-4, 1, 30)

        for lmbd in self.lambdas:
            loss_per_iter = []
            coef_per_iter = []

            w = np.zeros(d) # Start with zero weights
            old_loss = self._loss(X, y, w, lmbd) # Initial cost

            for it in range(self.max_iter):
                probs = self._sigmoid(X @ w) # Predictions
                resid = y - probs # Residuals (gradient direction)

                # Update each coordinate (feature) one at a time
                for j in range(d):
                    x_j = X[:, j] # j-th feature
                    grad = np.dot(x_j, resid) / n # Gradient w.r.t. w_j
                    soft = np.sign(grad) * max(abs(grad) - lmbd, 0) # Soft-thresholding
                    denom = np.sum(x_j ** 2) # Denominator for update
                    w[j] = soft / denom if denom != 0 else 0 # Coordinate update

                if it % 5 == 0:
                    # Recalculate residuals every few steps (lazy update)
                    resid = y - self._sigmoid(X @ w)

                new_loss = self._loss(X, y, w, lmbd) # Check new cost
                
                loss_per_iter.append(new_loss)
                coef_per_iter.append(w.copy())

                if abs(old_loss - new_loss) < self.tol:
                    # Stop if cost doesn't change much
                    break
                old_loss = new_loss

            self.coefs[lmbd] = w.copy()  # Save weights for this lambda
            self.training_history[lmbd] = {
                "loss": loss_per_iter,
                "coefs": coef_per_iter
            }

    def predict_proba(self, X, lmbd=None):
        """
        Predicts probabilities for input samples using trained model.

        Parameters:
        - X: input data
        - lmbd: lambda value to use (optional, defaults to last one)

        Returns:
        - vector of predicted probabilities (between 0 and 1)
        """
        X = np.c_[np.ones(X.shape[0]), X] # Add bias
        if lmbd is None:
            lmbd = list(self.coefs.keys())[-1] # Use last lambda if not specified

        # Use the closest available lambda (in case of float precision)
        closest = min(self.coefs.keys(), key=lambda l: abs(l - lmbd))
        w = self.coefs[closest] # Get corresponding weights
        return self._sigmoid(X @ w) # Compute probabilities

    def validate(self, X_val, y_val, metric='roc_auc'):
        """
        Evaluates the model on validation data and selects the best lambda.

        Parameters:
        - X_val: validation features
        - y_val: validation labels
        - metric: which metric to use ('roc_auc', 'f1', 'recall', etc.)

        Returns:
        - best lambda and corresponding score
        """
        X_val = np.c_[np.ones(X_val.shape[0]), X_val] # Add bias
        best_lambda = None
        best_score = -np.inf

        for lmbd, w in self.coefs.items():
            probs = self._sigmoid(X_val @ w) # Get probabilities

            # Choose evaluation metric
            if metric == 'roc_auc':
                score = roc_auc_score(y_val, probs)
            elif metric == 'pr_auc':
                score = average_precision_score(y_val, probs)
            else:
                preds = (probs >= 0.5).astype(int) # Convert to 0/1
                if metric == 'f1':
                    score = f1_score(y_val, preds)
                elif metric == 'recall':
                    score = recall_score(y_val, preds)
                elif metric == 'precision':
                    score = precision_score(y_val, preds)
                elif metric == 'balanced_accuracy':
                    score = balanced_accuracy_score(y_val, preds)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                # Update best result
                best_score = score
                best_lambda = lmbd

        return best_lambda, best_score

    def plot(self, X_val, y_val, metric='roc_auc'):
        """
        Plots how the evaluation metric changes with lambda.

        Parameters:
        - X_val: validation data
        - y_val: validation labels
        - metric: metric to evaluate (same options as in validate)
        """
        scores = []

        for lmbd in self.lambdas:
            probs = self.predict_proba(X_val, lmbd=lmbd)

            if metric == 'roc_auc':
                score = roc_auc_score(y_val, probs)
            elif metric == 'pr_auc':
                score = average_precision_score(y_val, probs)
            else:
                preds = (probs >= 0.5).astype(int)
                if metric == 'f1':
                    score = f1_score(y_val, preds)
                elif metric == 'recall':
                    score = recall_score(y_val, preds)
                elif metric == 'precision':
                    score = precision_score(y_val, preds)
                elif metric == 'balanced_accuracy':
                    score = balanced_accuracy_score(y_val, preds)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)

        plt.figure(figsize=(8, 5))
        plt.plot(self.lambdas, scores, marker='o')
        plt.xscale('log')
        plt.xlabel("Lambda")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} vs Lambda")
        plt.grid(True)
        plt.show()

    def plot_coefficients(self):
        """
        Plots how each weight (coefficient) changes across lambda values.

        Bias is shown separately as 'Bias', others as 'Feature i'.
        """
        plt.figure(figsize=(8, 5))
        num_features = len(next(iter(self.coefs.values())))

        for i in range(num_features):
            path = [w[i] for w in self.coefs.values()]
            label = "Bias" if i == 0 else f"Feature {i}"
            plt.plot(self.lambdas, path, label=label)

        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficient Value')
        plt.title('Coefficient Paths (Lasso)')
        plt.legend()
        plt.grid(True)
        plt.show()