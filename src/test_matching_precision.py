from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CSV_DIR = ROOT / "dataset" / "After marriage Data and separate to train and test"
XML_DIR = ROOT / "dataset" / "DataSet original"

FEATURES_FILE = CSV_DIR / "TrainDataForBoth.csv"
TEST_FILE = CSV_DIR / "TestDataForBoth.csv"

train = pd.read_csv(FEATURES_FILE, sep=";")
test = pd.read_csv(TEST_FILE, sep=";")

feature_columns = list(train.columns[:-1])

train_values = train[feature_columns].to_numpy(dtype=float)
test_values = test[feature_columns].to_numpy(dtype=float)

all_csv_values = np.vstack([train_values, test_values])

# Build lookup sets at different decimal precisions.
precisions = [3, 4, 5, 6, 7]

lookup_sets = {
    p: {
        tuple(row)
        for row in np.round(all_csv_values, p)
    }
    for p in precisions
}

results = {p: 0 for p in precisions}
total_frames = 0

xml_files = list(XML_DIR.rglob("*.xml"))

for xml_file in xml_files:
    root = ET.parse(xml_file).getroot()

    for frame in root:
        values = {
            measurement.tag: float(measurement.text)
            for measurement in frame
        }

        vector = np.array(
            [values[column] for column in feature_columns]
        )

        total_frames += 1

        for p in precisions:
            key = tuple(np.round(vector, p))

            if key in lookup_sets[p]:
                results[p] += 1

print("\n========== PRECISION COMPARISON ==========")
print("Total XML frames:", total_frames)

for p in precisions:
    matched = results[p]
    percentage = matched / total_frames * 100

    print(
        f"Precision: {p} decimals | "
        f"Matched: {matched}/{total_frames} | "
        f"Match rate: {percentage:.2f}%"
    )