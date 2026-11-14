from pathlib import Path
import numpy as np
import pandas as pd

DATASET_PATH = Path(
    "dataset/After marriage Data and separate to train and test"
)

ANGLE_COLUMNS = [
    "RightShoulderAngle",
    "RightElbowAngle",
    "RightWristAngle",
    "LeftShoulderAngle",
    "LeftElbowAngle",
    "LeftWristAngle",
]

LABEL_COLUMN = "Gool"


def load_data(split, data_type):
    file_path = DATASET_PATH / f"{split}DataFor{data_type}.csv"
    return pd.read_csv(file_path, sep=";")


for split in ["Train", "Test"]:
    print(f"\n{'=' * 65}")
    print(f"{split.upper()} DATASET RELATIONSHIP CHECK")
    print("=" * 65)

    angles = load_data(split, "Angles")
    points = load_data(split, "Point")
    both = load_data(split, "Both")

    # 1. Check row counts
    print("\n1. ROW COUNT CHECK")
    print("Angles:", len(angles))
    print("Points:", len(points))
    print("Both:", len(both))

    # 2. Check label alignment
    print("\n2. LABEL ALIGNMENT")
    print(
        "Angles vs Points:",
        angles[LABEL_COLUMN].equals(points[LABEL_COLUMN])
    )
    print(
        "Angles vs Both:",
        angles[LABEL_COLUMN].equals(both[LABEL_COLUMN])
    )

    # 3. Check angle measurements
    print("\n3. ANGLE FEATURE ALIGNMENT")
    print(
        "Angles vs Both:",
        np.allclose(
            angles[ANGLE_COLUMNS].to_numpy(),
            both[ANGLE_COLUMNS].to_numpy(),
            rtol=1e-6,
            atol=1e-8
        )
    )

    # 4. Check coordinate measurements
    point_columns = [
        column for column in points.columns
        if column != LABEL_COLUMN
    ]

    print("\n4. POINT FEATURE ALIGNMENT")
    print(
        "Points vs Both:",
        np.allclose(
            points[point_columns].to_numpy(),
            both[point_columns].to_numpy(),
            rtol=1e-6,
            atol=1e-8
        )
    )

# 5. Check whether train and test share exact feature rows
print(f"\n{'=' * 65}")
print("TRAIN-TEST OVERLAP CHECK")
print("=" * 65)

for data_type in ["Angles", "Point", "Both"]:
    train = load_data("Train", data_type)
    test = load_data("Test", data_type)

    train_features = train.drop(columns=[LABEL_COLUMN])
    test_features = test.drop(columns=[LABEL_COLUMN])

    train_rows = set(
        map(tuple, train_features.to_numpy())
    )
    test_rows = set(
        map(tuple, test_features.to_numpy())
    )

    overlap = train_rows.intersection(test_rows)

    print(f"{data_type}: {len(overlap)} shared feature rows")