# VYRONEX ML Datasets Directory

This directory stores real human participant dataset records collected via `ml/collect.py` according to `ml/protocol.md` and `ml/consent_form.md`.

## Contents
- `collected_features.csv`: Primary aggregated dataset with 12 columns:
  ```csv
  person_id,exercise,camera_angle,knee_angle,hip_angle,torso_angle,knee_alignment,symmetry,movement_speed,rep_duration,label,fault_type
  ```
- `landmarks/`: (Optional) Anonymized per-session landmark coordinate JSON/NPZ files storing 33-point body skeleton sequences without raw video imagery.

## Privacy Note
All data points are tagged by anonymous Participant ID (`P001`, `P002`, ...). No personally identifying information or raw facial video data is stored in this repository.
