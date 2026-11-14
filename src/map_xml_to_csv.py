
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XML_ROOT = PROJECT_ROOT / "dataset" / "DataSet original"

CSV_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "After marriage Data and separate to train and test"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD COMBINED CSV DATA
# --------------------------------------------------

train = pd.read_csv(CSV_ROOT / "TrainDataForBoth.csv", sep=";")
test = pd.read_csv(CSV_ROOT / "TestDataForBoth.csv", sep=";")

feature_columns = [
    column for column in train.columns
    if column != "Gool"
]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Number of features:", len(feature_columns))


# --------------------------------------------------
# 3. BUILD A LOOKUP DICTIONARY
# --------------------------------------------------

# Each rounded feature vector points to its CSV split and label.

lookup = {}

for split_name, dataframe in [
    ("Train", train),
    ("Test", test)
]:
    for row in dataframe.itertuples(index=False, name=None):
        values = np.array(row[:-1], dtype=float)
        label = int(row[-1])

        key = tuple(np.round(values, 5))

        lookup.setdefault(key, []).append(
            (split_name, label)
        )


# --------------------------------------------------
# 4. MATCH XML FRAMES AGAINST CSV ROWS
# --------------------------------------------------

xml_files = sorted(XML_ROOT.rglob("*.xml"))

results = []

for file_number, xml_file in enumerate(xml_files, start=1):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    matched_frames = 0
    unmatched_frames = 0

    recording_labels = set()
    recording_splits = set()

    for frame in root:

        frame_values = {
            measurement.tag: float(measurement.text)
            for measurement in frame
        }

        values = np.array(
            [frame_values[column] for column in feature_columns],
            dtype=float
        )

        key = tuple(np.round(values, 5))

        matches = lookup.get(key, [])

        if matches:
            matched_frames += 1

            for split_name, label in matches:
                recording_splits.add(split_name)
                recording_labels.add(label)
        else:
            unmatched_frames += 1

    results.append({
        "participant": xml_file.parent.name,
        "xml_file": xml_file.name,
        "matched_frames": matched_frames,
        "unmatched_frames": unmatched_frames,
        "labels_found": ",".join(
            map(str, sorted(recording_labels))
        ),
        "splits_found": ",".join(
            sorted(recording_splits)
        ),
        "single_label": len(recording_labels) == 1,
        "single_split": len(recording_splits) == 1,
        "complete_match": unmatched_frames == 0
    })

    if file_number % 100 == 0:
        print(f"Processed {file_number}/{len(xml_files)} XML files")


# --------------------------------------------------
# 5. SAVE AND DISPLAY RESULTS
# --------------------------------------------------

manifest = pd.DataFrame(results)

output_file = RESULTS_DIR / "xml_csv_mapping.csv"
manifest.to_csv(output_file, index=False)

print("\n========== XML-CSV MAPPING SUMMARY ==========")

print("Total XML recordings:", len(manifest))

print(
    "Complete recordings matched:",
    manifest["complete_match"].sum()
)

print(
    "Recordings with one label:",
    manifest["single_label"].sum()
)

print(
    "Recordings with one split:",
    manifest["single_split"].sum()
)

print("\nLabels found:")
print(manifest["labels_found"].value_counts().head(15))

print("\nSplits found:")
print(manifest["splits_found"].value_counts())

print("\nFirst 10 mappings:")
print(manifest.head(10).to_string(index=False))

print("\nSaved mapping to:", output_file)
