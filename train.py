import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from utils import (
    build_preprocessor,
    get_feature_columns_from_preprocessor,
    prepare_training_dataset,
    save_artifacts,
    save_training_metadata,
)


FEATURES = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience",
    "Location",
    "Company Type",
    "Industry",
]


def rupee(x):
    return f"Rs.{x:,.0f}"


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Model: {name}")
    print(f"  R2: {r2:.4f}")
    print(f"  MAE: {rupee(mae)}")
    print(f"  RMSE: {rupee(rmse)}")
    return {"name": name, "r2": r2, "mae": mae, "rmse": rmse, "preds": preds}


def tune_model(name, estimator, param_grid, X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    print(f"Tuning {name} with GridSearchCV...")
    search.fit(X_train, y_train)
    print(f"{name} best params: {search.best_params_}")
    return search.best_estimator_


def resolve_dataset_paths(root):
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    enhanced_csv = os.path.join(data_dir, "enhanced_salary_data.csv")
    candidates = [
        (os.path.join(data_dir, "real_india_salary_dataset.csv"), "real"),
        (os.path.join(root, "Salary_Data.csv"), "synthetic"),
        (os.path.join(data_dir, "updated_salary_dataset.csv"), "synthetic"),
    ]
    for path, source_type in candidates:
        if os.path.exists(path):
            return path, enhanced_csv, source_type
    raise FileNotFoundError("No source salary CSV found. Add data/real_india_salary_dataset.csv or data/updated_salary_dataset.csv.")


def main():
    root = os.path.dirname(__file__)
    raw_csv, prepared_csv, source_type = resolve_dataset_paths(root)

    print(f"Preparing dataset from {os.path.basename(raw_csv)} ({source_type})...")
    df = prepare_training_dataset(raw_csv, prepared_csv, source_type=source_type)

    X_df = df[FEATURES]
    y = df["Salary"].values
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    lr = tune_model(
        "LinearRegression",
        LinearRegression(),
        {"model__fit_intercept": [True, False], "model__positive": [False, True]},
        X_train_df,
        y_train,
    )
    dt = tune_model(
        "DecisionTree",
        DecisionTreeRegressor(random_state=42),
        {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        X_train_df,
        y_train,
    )
    rf_best = tune_model(
        "RandomForest",
        RandomForestRegressor(random_state=42),
        {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        X_train_df,
        y_train,
    )

    print("\nEvaluating models:")
    results = [
        evaluate_model("LinearRegression", lr, X_test_df, y_test),
        evaluate_model("DecisionTree", dt, X_test_df, y_test),
        evaluate_model("RandomForest", rf_best, X_test_df, y_test),
    ]

    best = min(results, key=lambda r: r["rmse"])
    print(f"\nSelected best model: {best['name']} (RMSE {rupee(best['rmse'])})")
    best_model = {"LinearRegression": lr, "DecisionTree": dt, "RandomForest": rf_best}[best["name"]]
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    feature_columns = get_feature_columns_from_preprocessor(fitted_preprocessor)

    model_folder = os.path.join(root, "model")
    os.makedirs(model_folder, exist_ok=True)
    save_artifacts(best_model, columns=feature_columns, folder=model_folder)
    joblib.dump(fitted_preprocessor.named_transformers_["cat"], os.path.join(model_folder, "encoder.pkl"))
    joblib.dump(fitted_preprocessor.named_transformers_["num"], os.path.join(model_folder, "scaler.pkl"))
    save_training_metadata(
        model_folder,
        {
            "source_file": os.path.basename(raw_csv),
            "source_type": source_type,
            "prepared_file": os.path.basename(prepared_csv),
            "rows": int(len(df)),
            "features": FEATURES,
            "best_model": best["name"],
            "rmse": float(best["rmse"]),
        },
    )

    plots_dir = os.path.join(root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    preds = best_model.predict(X_test_df)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Salary (INR)")
    plt.ylabel("Predicted Salary (INR)")
    plt.title("Predicted vs Actual Salaries")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"))
    plt.close()

    try:
        importances = rf_best.named_steps["model"].feature_importances_
        feat_names = get_feature_columns_from_preprocessor(rf_best.named_steps["preprocessor"])
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)[:20]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=fi.values, y=fi.index)
        plt.title("Feature Importance (RandomForest)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
        plt.close()
    except Exception:
        print("Feature importance not available for selected model.")

    comp = pd.DataFrame(results).drop(columns=["preds"])
    comp = comp[["name", "r2", "mae", "rmse"]]
    comp.to_csv(os.path.join(plots_dir, "model_comparison.csv"), index=False)
    print("Training complete. Artifacts saved to model/ and plots/.")


if __name__ == "__main__":
    main()
