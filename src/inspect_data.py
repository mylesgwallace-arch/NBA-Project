import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

for file in RAW.iterdir():
    print("\n" + "=" * 80)
    print(file.name)
    print("=" * 80)

    if file.suffix == ".csv":
        df = pd.read_csv(file)
    elif file.suffix == ".parquet":
        df = pd.read_parquet(file)
    else:
        continue

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}: {df[column].dtype}")

    print("\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).head(10))

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))