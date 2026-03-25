import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = os.path.dirname(__file__)
DATASET_PATH = os.path.join(ROOT, "data", "real_india_salary_dataset.csv")
IMAGES_DIR = os.path.join(ROOT, "images")


def ensure_images_dir():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def save_plot(filename):
    path = os.path.join(IMAGES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top10_jobs(df):
    top_jobs = df.groupby("Job Title")["Salary"].mean().nlargest(10).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="viridis")
    plt.title("Top 10 Highest Paying Job Roles")
    plt.xlabel("Average Salary (INR)")
    plt.ylabel("Job Title")
    save_plot("Top10.png")


def plot_distribution(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    sns.histplot(df["Age"], bins=25, kde=True, ax=axes[0], color="#3b82f6")
    axes[0].set_title("Age Distribution")
    sns.histplot(df["Years of Experience"], bins=25, kde=True, ax=axes[1], color="#f59e0b")
    axes[1].set_title("Years of Experience Distribution")
    sns.histplot(df["Salary"], bins=30, kde=True, ax=axes[2], color="#10b981")
    axes[2].set_title("Salary Distribution")
    save_plot("Distribution.png")


def plot_education_gender_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x="Gender", order=df["Gender"].value_counts().index, ax=axes[0], palette="magma")
    axes[0].set_title("Gender Distribution")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Count")

    edu_order = df["Education Level"].value_counts().index
    sns.countplot(data=df, x="Education Level", order=edu_order, ax=axes[1], palette="rocket")
    axes[1].set_title("Education Level Distribution")
    axes[1].set_xlabel("Education Level")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=35)
    save_plot("ed&gender_distribution.png")


def plot_education_salary_gender(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Education Level",
        y="Salary",
        hue="Gender",
        estimator="median",
        errorbar=None,
        palette="Set2",
    )
    plt.title("Education Level vs Salary by Gender")
    plt.xlabel("Education Level")
    plt.ylabel("Median Salary (INR)")
    plt.xticks(rotation=35)
    save_plot("ed_salary_gender.png")


def plot_heatmap(df):
    temp = df.copy()
    for col in ["Gender", "Education Level", "Job Title", "Location", "Company Type", "Industry"]:
        if col in temp.columns:
            temp[col] = temp[col].astype("category").cat.codes

    num_cols = [c for c in temp.columns if temp[c].dtype != object]
    corr = temp[num_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    save_plot("Heatmap.png")


def plot_correlation_salary(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(data=df, x="Age", y="Salary", ax=axes[0], scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "red"})
    axes[0].set_title("Age vs Salary")
    sns.regplot(
        data=df,
        x="Years of Experience",
        y="Salary",
        ax=axes[1],
        scatter_kws={"alpha": 0.35, "s": 18},
        line_kws={"color": "red"},
    )
    axes[1].set_title("Experience vs Salary")
    save_plot("Correlation.png")


def plot_feature_importance_copy():
    source = os.path.join(ROOT, "plots", "feature_importance.png")
    target = os.path.join(IMAGES_DIR, "Feature_Imp.png")
    if os.path.exists(source):
        img = plt.imread(source)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        save_plot("Feature_Imp.png")


def remove_stale_files():
    stale = ["dummy.md"]
    for name in stale:
        path = os.path.join(IMAGES_DIR, name)
        if os.path.exists(path):
            os.remove(path)


def main():
    ensure_images_dir()
    df = load_dataset()

    # Standardize numeric columns defensively.
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Years of Experience"] = pd.to_numeric(df["Years of Experience"], errors="coerce")
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Age", "Years of Experience", "Salary", "Job Title", "Education Level", "Gender"])

    sns.set_theme(style="whitegrid")
    plot_top10_jobs(df)
    plot_distribution(df)
    plot_education_gender_distribution(df)
    plot_education_salary_gender(df)
    plot_heatmap(df)
    plot_correlation_salary(df)
    plot_feature_importance_copy()
    remove_stale_files()
    print("EDA images regenerated in images/.")


if __name__ == "__main__":
    main()
