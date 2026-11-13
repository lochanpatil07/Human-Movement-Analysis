
import xml.etree.ElementTree as ET
from pathlib import Path

# Locate the dataset
DATASET_PATH = Path("dataset/DataSet original")

# Find the first XML recording automatically
xml_files = sorted(DATASET_PATH.rglob("*.xml"))

if not xml_files:
    raise FileNotFoundError("No XML files found!")

sample_file = xml_files[0]

print("SAMPLE XML FILE")
print(sample_file)
print("-" * 50)

# Parse XML
tree = ET.parse(sample_file)
root = tree.getroot()

print("Root tag:", root.tag)
print("Total frames:", len(root))

# Inspect the first frame
first_frame = root[0]

print("\nFIRST FRAME")
print("Frame tag:", first_frame.tag)
print("Number of measurements:", len(first_frame))

print("\nMEASUREMENTS IN FIRST FRAME")

for measurement in first_frame:
    print(f"{measurement.tag}: {measurement.text}")

# Inspect frame structure consistency
measurement_counts = {
    len(frame) for frame in root
}

print("\nSTRUCTURE CHECK")
print("Unique measurement counts across frames:",
      measurement_counts)
