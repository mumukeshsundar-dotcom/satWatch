
from pathlib import Path

import pandas as pd

# Locate the dataset relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"

print(f"Dataset path: {DATASET_PATH}")

# Load the CSV.
df = pd.read_csv(DATASET_PATH)

# Basic structure.
print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head().to_string())

print("\n=== DATA TYPES ===")
print(df.dtypes.to_string())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sort_values(ascending=False).to_string())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== NUMERICAL SUMMARY ===")
print(df.describe().to_string())

print("\n=== POSSIBLE LABEL / SPLIT COLUMNS ===")
for column in df.columns:
    if any(
        keyword in column.lower()
        for keyword in ("label", "anomaly", "class", "train", "split", "target")
    ):
        print(f"\nColumn: {column}")
        print(df[column].value_counts(dropna=False).head(20).to_string())
