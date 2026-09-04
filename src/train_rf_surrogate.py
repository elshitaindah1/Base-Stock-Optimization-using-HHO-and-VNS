import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load CSV data, remove repeated headers, and drop missing values."""
    df = pd.read_csv(file_path)

    # Expected columns for ATO base-stock surrogate model
    expected_columns = [f"x{i}" for i in range(1, 13)] + ["profit"]

    # Verify required columns are present
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Coerce the expected columns to numeric values.
    # Repeated header rows like 'x1' become NaN and can be dropped.
    df[expected_columns] = df[expected_columns].apply(pd.to_numeric, errors="coerce")

    # Drop rows containing missing values in the expected columns.
    before_drop = len(df)
    df = df.dropna(subset=expected_columns)
    after_drop = len(df)
    dropped = before_drop - after_drop
    print(f"Loaded {before_drop} rows, dropped {dropped} rows with repeated headers or missing values.")

    return df


def build_random_forest_model(random_state: int = 42) -> RandomForestRegressor:
    """Create a RandomForestRegressor with a fixed random state."""
    return RandomForestRegressor(random_state=random_state, n_jobs=-1)


def main(data_file: str) -> None:
    """Main routine to train and evaluate the surrogate random forest model."""
    # 1. Load and clean dataset
    df = load_and_clean_data(data_file)

    # 2. Prepare features and target
    feature_columns = [f"x{i}" for i in range(1, 13)]
    target_column = "profit"
    X = df[feature_columns].values
    y = df[target_column].values

    # 3. Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 4. Setup hyperparameter search space for RandomizedSearchCV
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    rf = build_random_forest_model(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="r2",
        cv=5,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    # 5. Fit RandomizedSearchCV on the training data
    random_search.fit(X_train, y_train)

    print("Best hyperparameters found:")
    print(random_search.best_params_)
    print(f"Best CV R2: {random_search.best_score_:.4f}")

    # 6. Evaluate best model on the test set
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"Test R2: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # 7. Display feature importance
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values(by="importance", ascending=False)
    print("Feature importances:")
    print(importance_df.to_string(index=False))

    # 8. Generate Actual vs Predicted scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Actual vs Predicted Profit - Random Forest Surrogate")
    plt.grid(True)
    plt.tight_layout()
    plot_filename = "actual_vs_predicted.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Scatter plot saved to: {plot_filename}")

    # 9. Save the best model to disk using joblib
    model_filename = "best_random_forest_surrogate.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Best model saved to: {model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest surrogate model for ATO base-stock levels."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="training_data.csv",
        help="Path to the input CSV dataset file.",
    )
    args = parser.parse_args()
    main(args.data_file)
