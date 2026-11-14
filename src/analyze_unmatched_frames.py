
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

CSV_DIR = ROOT / "dataset" / "After marriage Data and separate to train and test"
XML_DIR = ROOT / "dataset" / "DataSet original"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD TRAINING AND TESTING DATA
# --------------------------------------------------

train = pd.read_csv(
    CSV_DIR / "TrainDataForBoth.csv",
    sep=";"
)

test = pd.read_csv(
    CSV_DIR / "TestDataForBoth.csv",
    sep=";"
)

feature_columns = list(train.columns[:-1])

train["split"] = "Train"
test["split"] = "Test"

csv_data = pd.concat(
    [train, test],
    ignore_index=True
)

csv_features = csv_data[feature_columns].to_numpy(dtype=float)

labels = csv_data["Gool"].to_numpy()
splits = csv_data["split"].to_numpy()

print("CSV observations:", len(csv_data))
print("Number of features:", len(feature_columns))


# --------------------------------------------------
# 3. BUILD NEAREST-NEIGHBOUR SEARCH TREE
# --------------------------------------------------

tree = cKDTree(csv_features)


# --------------------------------------------------
# 4. COMPARE EVERY XML FRAME WITH CSV DATA
# --------------------------------------------------

results = []

xml_files = sorted(XML_DIR.rglob("*.xml"))

for file_number, xml_file in enumerate(xml_files, start=1):

    root = ET.parse(xml_file).getroot()

    participant = xml_file.parent.name

    for frame in root:

        values = {
            measurement.tag: float(measurement.text)
            for measurement in frame
        }

        xml_vector = np.array(
            [values[column] for column in feature_columns],
            dtype=float
        )

        # Find the nearest CSV observation.
        distance, nearest_index = tree.query(
            xml_vector,
            k=1
        )

        nearest_csv_vector = csv_features[nearest_index]

        differences = np.abs(
            xml_vector - nearest_csv_vector
        )

        results.append({
            "participant": participant,
            "xml_file": xml_file.name,
            "frame": frame.tag,
            "nearest_distance": distance,
            "rmse": np.sqrt(np.mean(differences ** 2)),
            "max_absolute_difference": differences.max(),
            "features_with_difference_below_0.001": np.sum(
                differences < 0.001
            ),
            "nearest_label": labels[nearest_index],
            "nearest_split": splits[nearest_index]
        })

    if file_number % 100 == 0:
        print(f"Processed {file_number}/{len(xml_files)} XML files")


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = RESULTS_DIR / "nearest_neighbor_analysis.csv"

results_df.to_csv(output_file, index=False)


# --------------------------------------------------
# 6. ANALYZE DISTANCE DISTRIBUTION
# --------------------------------------------------

print("\n========== NEAREST-NEIGHBOUR ANALYSIS ==========")

print("Total XML frames:", len(results_df))

print("\nNearest distance statistics:")
print(results_df["nearest_distance"].describe())

print("\nMaximum absolute difference statistics:")
print(results_df["max_absolute_difference"].describe())

print("\nRMSE statistics:")
print(results_df["rmse"].describe())


# --------------------------------------------------
# 7. INVESTIGATE DIFFERENT DISTANCE THRESHOLDS
# --------------------------------------------------

print("\n========== DISTANCE THRESHOLD ANALYSIS ==========")

for threshold in [0.0001, 0.001, 0.01, 0.1, 1.0]:

    matched = (
        results_df["max_absolute_difference"] <= threshold
    ).sum()

    percentage = matched / len(results_df) * 100

    print(
        f"Maximum difference <= {threshold}: "
        f"{matched}/{len(results_df)} "
        f"({percentage:.2f}%)"
    )


# --------------------------------------------------
# 8. SHOW THE MOST DIFFERENT OBSERVATIONS
# --------------------------------------------------

print("\n========== 20 LARGEST DIFFERENCES ==========")

print(
    results_df.nlargest(
        20,
        "max_absolute_difference"
    )[
        [
            "participant",
            "xml_file",
            "frame",
            "nearest_distance",
            "max_absolute_difference",
            "nearest_label",
            "nearest_split"
        ]
    ].to_string(index=False)
)

print("\nSaved analysis to:", output_file)
