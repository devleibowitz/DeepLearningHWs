import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score
import numpy as np
from sklearn.base import BaseEstimator
import joblib


# Custom classifier wrapper for GridSearchCV compatibility
class LGBMWrapper(BaseEstimator):
    def __init__(self, num_leaves=31, learning_rate=0.05, max_depth=-1, n_estimators=100, scale_pos_weight=1):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.model = None

    def fit(self, X, y):
        params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'scale_pos_weight': self.scale_pos_weight
        }

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

# Function to tune hyperparameters and threshold
def tune_lightgbm(data_path):
    # Load the data
    df = pd.read_csv(data_path)
    
    # Clean feature names by replacing special characters with underscores
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

    # Assuming 'label' column is the target
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a custom LightGBM model with parameter tuning wrapper
    lgbm = LGBMWrapper()

    # Define hyperparameters grid for tuning
    param_grid = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [-1, 5, 10],
        'n_estimators': [100, 200, 300],
        'scale_pos_weight': [1, 2, 3]
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, scoring='recall', verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Predict probabilities on the test data
    y_pred_prob = best_model.predict_proba(X_test)

    # Find the best threshold based on maximizing recall for class 1
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_recall = 0
    best_threshold = 0

    for threshold in thresholds:
        y_pred = (y_pred_prob > threshold).astype(int)
        recall = recall_score(y_test, y_pred)
        if recall > best_recall:
            best_recall = recall
            best_threshold = threshold

    print(f"Best threshold for maximizing recall: {best_threshold}")
    
    # Apply the best threshold to make final predictions
    y_pred = (y_pred_prob > best_threshold).astype(int)

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Optionally, save the best model
    joblib.dump(best_model, 'lightgbm_best_model.joblib')

def train_lightgbm_with_best_params(data_path):
    df = pd.read_csv(data_path)
    
    # Clean feature names by replacing special characters with underscores
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

    # Assuming 'label' column is the target
    X = df.drop(columns=['label', 'Unnamed__0', 'id'])
    y = df['label']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Best parameters from your tuning
    best_params = {
        'learning_rate': 0.01,
        'max_depth': 5,
        'n_estimators': 300,
        'num_leaves': 31,
        'scale_pos_weight': 3  # Adjusted for class imbalance
    }
    
    # Create the LightGBM classifier with the best parameters
    model = lgb.LGBMClassifier(**best_params)

    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Find the optimal threshold for recall maximization
    optimal_threshold = .2

    # Make predictions using the optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print the recall score for class 1 (positive class)
    print(f"Recall for class 1 (positive class): {recall_score(y_test, y_pred, pos_label=1)}")

    # Plot and show feature importance
    import matplotlib.pyplot as plt
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', figsize=(10, 6))
    plt.title("Feature Importance by Gain")
    plt.tight_layout()
    plt.show()

    
    # Save the trained model
    joblib.dump(model, 'lightgbm_best_model.joblib')

# Main function to run the script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a LightGBM model on EHR data and tune parameters.')
    parser.add_argument('data_path', type=str, help='Path to the labeled EHR dataset')
    args = parser.parse_args()
    
    # Tune the model with the provided data
    train_lightgbm_with_best_params(args.data_path)