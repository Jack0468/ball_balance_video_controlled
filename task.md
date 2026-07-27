# Data Architecture & VLA Collection Tasks

- `[x]` 1. Data Centralization
  - `[x]` Create `host_software/data`
  - `[x]` Move `ml_vision/data/01_bronze`, `02_silver`, `03_synthetic_yolo`, etc. to `host_software/data`
  - `[x]` Update all data paths in python scripts

- `[x]` 2. VLA Data Collection Pipeline
  - `[x]` Write `host_software/data_collection/collect_vla_data.py`
  - `[x]` Rewrite `host_software/ml_multimodal/data_processing/generate_vla_dataset.py`

- `[x]` 3. Documentation & Catalog
  - `[x]` Write `host_software/data_collection/IPHONE_PIPELINE.md`
  - `[x]` Write `host_software/data_collection/DATASET_CATALOG.md`
