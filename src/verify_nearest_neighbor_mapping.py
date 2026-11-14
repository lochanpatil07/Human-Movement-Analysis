
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MAPPING_FILE = ROOT / "results" / "nearest_neighbor_analysis.csv"
SPLIT_FILE = ROOT / "results" / "xml_csv_mapping.csv"

nearest = pd.read_csv(MAPPING_FILE)
original = pd.read_csv(SPLIT_FILE)

# Expected numeric label comes from Data1.xml ... Data8.xml.
nearest["expected_label"] = (
    nearest["xml_file"]
    .str.extract(r"Data(\d+)\.xml")[0]
    .astype(int)
)

# Recover the participant's verified split.
participant_splits = (
    original[["participant", "splits_found"]]
    .drop_duplicates()
)

nearest = nearest.merge(
    participant_splits,
    on="participant",
    how="left"
)

nearest["label_correct"] = (
    nearest["nearest_label"] == nearest["expected_label"]
)

nearest["split_correct"] = (
    nearest["nearest_split"] == nearest["splits_found"]
)

print("\n========== NEAREST-NEIGHBOUR VERIFICATION ==========")

print("\nLabel agreement:")
print(nearest["label_correct"].value_counts())

print("\nSplit agreement:")
print(nearest["split_correct"].value_counts())

print("\nLabel and split both correct:")
print(
    (
        nearest["label_correct"]
        & nearest["split_correct"]
    ).value_counts()
)

print("\n========== INCORRECT LABEL MATCHES ==========")

wrong_labels = nearest[~nearest["label_correct"]]

print("Incorrect label matches:", len(wrong_labels))

print(
    wrong_labels[
        [
            "participant",
            "xml_file",
            "frame",
            "expected_label",
            "nearest_label",
            "nearest_distance"
        ]
    ].head(20).to_string(index=False)
)

print("\n========== INCORRECT SPLIT MATCHES ==========")

wrong_splits = nearest[~nearest["split_correct"]]

print("Incorrect split matches:", len(wrong_splits))

print(
    wrong_splits[
        [
            "participant",
            "xml_file",
            "frame",
            "splits_found",
            "nearest_split",
            "nearest_distance"
        ]
    ].head(20).to_string(index=False)
)

print("\n========== SUMMARY ==========")

print(
    "Both correct:",
    (
        nearest["label_correct"]
        & nearest["split_correct"]
    ).sum(),
    "/",
    len(nearest)
)
