
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_FILE = PROJECT_ROOT / "results" / "xml_csv_mapping.csv"

df = pd.read_csv(MAPPING_FILE)

print("\n========== PARTICIPANT SPLIT AUDIT ==========")

# Count recordings and distinct splits for each participant
participant_summary = (
    df.groupby("participant")
    .agg(
        recordings=("xml_file", "count"),
        number_of_splits=("splits_found", "nunique"),
        split=("splits_found", "first")
    )
    .reset_index()
)

print("\nParticipant count:", len(participant_summary))

print("\nRecordings per participant:")
print(participant_summary["recordings"].value_counts().sort_index())

print("\nParticipants by split:")
print(participant_summary["split"].value_counts(dropna=False))

print("\nParticipants appearing in multiple splits:")
mixed = participant_summary[
    participant_summary["number_of_splits"] > 1
]
print(mixed.to_string(index=False))

print("\nParticipants with missing or multiple split assignments:")
invalid = participant_summary[
    (participant_summary["number_of_splits"] != 1)
    | (participant_summary["split"].isna())
]
print(invalid.to_string(index=False))

# Check whether each XML filename consistently maps to one label
print("\n========== FILENAME-LABEL AUDIT ==========")

filename_summary = (
    df.groupby("xml_file")["labels_found"]
    .nunique()
)

print("Distinct labels per XML filename:")
print(filename_summary.value_counts().sort_index())

print("\nFilename to label mapping:")
print(
    df.groupby("xml_file")["labels_found"]
    .first()
    .to_string()
)

# Investigate incomplete frame matches
print("\n========== FRAME MATCH AUDIT ==========")

print("\nMatched-frame distribution:")
print(df["matched_frames"].value_counts().sort_index())

print("\nTotal matched frames:", df["matched_frames"].sum())
print("Total expected frames:", len(df) * 30)

print("\nRecordings with fewer than 30 matched frames:")
print(
    df.loc[
        df["matched_frames"] < 30,
        ["participant", "xml_file", "matched_frames",
         "unmatched_frames", "labels_found", "splits_found"]
    ].head(20).to_string(index=False)
)
