# VRI 2026 Dataset Catalog

This ledger tracks all physical data collected for the VRI 2026 project. All datasets are centralized in `host_software/data`.

## Legacy iPhone Datasets
*Collected using `iphone_data_logger.py` and synced via `sync_data.py`.*

- **Video 1**: Collected on 22nd July 2026. Purpose: PID tracking baseline.
- **Video 2**: Collected on 22nd July 2026. Purpose: PID tracking baseline.
- **Video 3**: Collected on 23rd July 2026. Purpose: PID tracking baseline.
- **Video 4**: Collected on 23rd July 2026. Purpose: PID tracking baseline.
- **Video 5**: Collected on 24th July 2026. Purpose: PID tracking baseline.

> [!CAUTION]
> The Legacy iPhone Datasets **DO NOT** contain semantic targets (like "go red"). They cannot be used to train the Vision-Language-Action (VLA) model's language bindings.

## Multimodal VLA Datasets
*Collected using `collect_vla_data.py`.*

- *(No VLA data collected yet. Run `collect_vla_data.py` to begin recording semantic demonstrations).*
