
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("dataset/DataSet original")
OUTPUT_PATH = Path("results/recording_manifest.csv")

xml_files = sorted(DATASET_PATH.rglob("*.xml"))

if not xml_files:
    raise FileNotFoundError(f"No XML files found under: {DATASET_PATH}")

results = []

for index, file_path in enumerate(xml_files, start=1):
    record = {
        "participant": file_path.parent.name,
        "recording": file_path.name,
        "frame_count": 0,
        "measurements_per_frame": None,
        "consistent_structure": False,
        "numeric_values": False,
        "parse_status": "OK",
        "error": ""
    }

    try:
        root = ET.parse(file_path).getroot()
        frames = list(root)

        record["frame_count"] = len(frames)

        if not frames:
            raise ValueError("Recording contains no frames")

        first_tags = {item.tag for item in frames[0]}
        measurement_counts = {len(frame) for frame in frames}

        record["measurements_per_frame"] = len(frames[0])
        record["consistent_structure"] = (
            len(measurement_counts) == 1
            and all({item.tag for item in frame} == first_tags
                    for frame in frames)
        )

        values_are_numeric = True

        for frame in frames:
            for measurement in frame:
                try:
                    float(measurement.text)
                except (TypeError, ValueError):
                    values_are_numeric = False
                    break
            if not values_are_numeric:
                break

        record["numeric_values"] = values_are_numeric

    except Exception as error:
        record["parse_status"] = "ERROR"
        record["error"] = str(error)

    results.append(record)

    if index % 100 == 0:
        print(f"Checked {index}/{len(xml_files)} recordings")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False)

print("\n========== DATASET VALIDATION SUMMARY ==========")
print("Total XML files:", len(df))
print("Successfully parsed:", (df["parse_status"] == "OK").sum())
print("Parsing errors:", (df["parse_status"] == "ERROR").sum())
print("Consistent structure:", df["consistent_structure"].sum())
print("Numeric values valid:", df["numeric_values"].sum())
print("Recordings with 51 measurements per frame:",
      (df["measurements_per_frame"] == 51).sum())

print("\nFrame-count distribution:")
print(df["frame_count"].value_counts().sort_index())

print("\nManifest saved to:", OUTPUT_PATH)
