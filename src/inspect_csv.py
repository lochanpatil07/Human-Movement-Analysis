
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("dataset")

csv_files = sorted(DATASET_PATH.rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files")

for file_path in csv_files:
    print("\n" + "=" * 70)
    print("FILE:", file_path.name)

    # Important: these CSV files use semicolons
    df = pd.read_csv(file_path, sep=";")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    if "Gool" in df.columns:
        print("\nGool value distribution:")
        print(df["Gool"].value_counts(dropna=False).sort_index())
