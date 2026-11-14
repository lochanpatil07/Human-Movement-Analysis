# Human Movement Analysis

## Project Overview

This project investigates human movement patterns using 3D skeletal data. It aims to study joint positions, joint angles, and variations in movement patterns across different participants performing predefined human poses.

The project uses Python for data inspection, preprocessing, analysis, and potential machine learning experiments.

## Research Objective

To investigate how 3D joint positions and joint-angle patterns vary across individuals performing the same predefined human poses.

## Dataset Description

The dataset contains recordings from 127 volunteers.

| Dataset property | Description |
|---|---|
| Participants | 127 |
| Recordings per participant | 8 |
| Total XML recordings | 1,016 |
| Frames per recording | 30 |
| Total frames | 30,480 |
| Measurements per frame | 51 |
| Spatial measurements | 45 (15 joints × 3 coordinates) |
| Angular measurements | 6 |

Each frame contains 3D camera coordinates for 15 body joints and six joint-angle measurements.

The dataset documentation describes eight predefined human positions:

1. Hands up
2. Right hand up
3. Left hand up
4. Hands-on head
5. Arms open
6. Stand up straight
7. Hands-on waist
8. Hands forward

The semantic correspondence between these descriptions and the numeric labels is still to be independently verified.

## Repository Structure

```text
Human-Movement-Analysis/
│
├── dataset/
│   └── README.md
│
├── src/
│   ├── inspect_xml.py
│   ├── validate_dataset.py
│   ├── inspect_csv.py
│   ├── audit_csv_relationships.py
│   ├── map_xml_to_csv.py
│   ├── audit_participant_split.py
│   └── test_matching_precision.py
│
├── results/
│   ├── recording_manifest.csv
│   └── xml_csv_mapping.csv
│
├── README.md
└── .gitignore
```

The original dataset files are excluded from version control.

## Dataset Validation

All 1,016 XML recordings were successfully parsed.

Validation results:

- No XML parsing errors.
- All recordings contain 30 frames.
- All frames contain 51 measurements.
- All measurement values are numeric.
- The XML structure is consistent across recordings.

## Training and Testing Split

The supplied CSV datasets contain:

| Split | Samples | Participants |
|---|---:|---:|
| Training | 22,080 | 92 |
| Testing | 8,400 | 35 |
| Total | 30,480 | 127 |

Participant-level auditing confirmed that no participant appears in both training and testing sets.

This provides a participant-independent split for future model evaluation.

## XML-to-CSV Relationship

The XML recordings were compared with the combined CSV feature dataset using their 51 numerical measurements.

The filename-to-label relationship was consistent across all participants:

- Data1.xml → Label 1
- Data2.xml → Label 2
- Data3.xml → Label 3
- Data4.xml → Label 4
- Data5.xml → Label 5
- Data6.xml → Label 6
- Data7.xml → Label 7
- Data8.xml → Label 8

At five-decimal rounding precision, 29,616 of 30,480 XML frames matched CSV observations (97.17%).

## Numerical Precision Experiment

Matching was tested at different decimal precisions.

| Decimal precision | Matched frames | Match rate |
|---:|---:|---:|
| 3 | 30,468 | 99.96% |
| 4 | 30,413 | 99.78% |
| 5 | 29,616 | 97.17% |
| 6 | 22,409 | 73.52% |
| 7 | 23,770 | 77.99% |

The matching rate varies with rounding precision. This indicates that numerical representation affects the matching process.

Further investigation is required to determine whether unmatched observations result from rounding differences, omitted frames, or other causes. Higher match rates at lower precision do not necessarily establish unique or correct matches.

## Current Progress

- [x] XML structure inspection
- [x] XML dataset validation
- [x] CSV dataset inspection
- [x] Train/test CSV relationship audit
- [x] XML-to-CSV label and split mapping
- [x] Participant-level split verification
- [x] Numerical precision sensitivity experiment
- [ ] Investigate unmatched XML frames
- [ ] Verify numeric labels against pose descriptions
- [ ] Perform exploratory data analysis
- [ ] Study inter-participant movement variability
- [ ] Develop and evaluate baseline models

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- XML ElementTree
- Scikit-learn (planned)

## Research Status

This project is currently in the dataset validation and exploratory analysis stage. Conclusions about movement patterns and model performance will be added after the corresponding experiments have been completed.