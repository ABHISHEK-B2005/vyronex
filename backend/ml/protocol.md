# VYRONEX Movement Data Collection Protocol (8A)

Standardized operating procedure for recording human movement datasets for training and benchmarking biomechanical injury-risk screening models.

---

## 1. Cohort Recruitment & Diversity Guidelines

### Target Sample Size
- **10 to 30 unique human participants**.
- Diversity of participants is prioritized over high repetition volume from a single subject.

### Demographic & Physical Variance Targets
To ensure model generalizability across diverse real-world users, recruit across:
1. **Background:**
   - General gym / fitness center members (recreational experience).
   - Collegiate varsity / club athletes.
   - Track-and-field / jumping / throwing athletes (specialized movement mechanics).
2. **Anthropometrics & Body Diversity:**
   - Varying heights (e.g., 150 cm to 200+ cm).
   - Varying body shapes, limb-to-torso ratios, and flexibility levels.
3. **Apparel & Visual Appearance:**
   - Varied clothing fits (fitted compression, regular t-shirts, loose athletic shorts, track pants).
   - High-contrast and low-contrast color variations against the gym backdrop.
   - Footwear: athletic sneakers, flat lifting shoes, and barefoot/socks (where safe).

---

## 2. Participant Privacy & De-Identification

1. **Anonymous Participant ID:**
   - Assign each participant an identifier strictly following the pattern `P001`, `P002`, `P003`, ..., `P030`.
   - Never write, store, or tag participant names or personal details in video file names, dataset CSVs, or landmark JSON files.
2. **Landmark-First Retention:**
   - Process camera frames into 33-point MediaPipe landmark coordinate streams.
   - Retain landmark coordinate files (`.json` / `.npz` / `.csv`) for ML dataset building.
   - Delete temporary raw video footage immediately after landmark verification unless explicit secondary written consent is provided.

---

## 3. Camera Setup & Environmental Specifications

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Camera Angles** | **Front (0°)**, **45-Degree Oblique**, **Side (90°)** | Position tripod securely at designated angles. |
| **Camera Distance** | **2.5 m – 3.5 m** | Entire body (top of head to below feet) must remain visible throughout the full range of motion. |
| **Camera Height** | **0.9 m – 1.1 m** (waist / mid-torso height) | Level with the subject's center of mass; no steep downwards/upwards pitch. |
| **Lighting** | **Minimum 300 lux**, diffuse illumination | Avoid harsh direct backlighting (e.g., windows behind the athlete) and dark shadows. |
| **Frame Rate** | **30 FPS or 60 FPS** | Constant frame rate; avoid variable frame-rate smartphone modes. |
| **Background** | Clean, uncluttered background | Minimize visual noise, reflections, and bystanders walking behind the lifter. |

---

## 4. Exercise Styles & Ground Truth Labels

Every participant performs sets of the target exercise (e.g. Squats) across three distinct, controlled movement styles:

| Style | Description & Execution | Fault Examples | Target ML Label | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| **Correct** | Proper biomechanical technique, balanced stance, neutral spine, controlled descent & ascent, hips below/parallel to knees. | None | `LOW` | Safe form |
| **Minor Error** | Slight, isolated deviation in one movement plane; general stability maintained. | Slight knee valgus (inward wobble), minor asymmetrical shift, slightly fast descent. | `MODERATE` | Mild defect |
| **Dangerous Form** | Noticeable major fault compromising spinal or joint alignment. | Severe knee collapse, excessive forward trunk lean / rounding of the lumbar spine, dramatic hip shift. | `HIGH` | High risk |

---

## 5. Participant Safety Protocol (MANDATORY)

> [!CAUTION]
> **SAFETY IS THE ABSOLUTE HIGHEST PRIORITY.**  
> Never ask any participant to perform an unsafe movement under heavy resistance.

1. **Unweighted / Bodyweight Only:**
   - All simulated "Minor Error" and "Dangerous Form" reps **MUST** be performed using bodyweight only or a featherweight plastic/PVC dowel (0 kg load).
   - Under no circumstances should weighted barbells, heavy dumbbells, or external loads be used during intentional fault simulation.
2. **Controlled Simulation:**
   - "Dangerous form" is simulated through slow, deliberate postural adjustments (e.g., gently demonstrating forward lean or rounded shoulders), never dynamic jerking or impact.
3. **Warm-Up & Rest:**
   - 3-5 minute mobility warm-up before recording.
   - At least 60 seconds of rest between sets to prevent fatigue-induced uncontrolled form breakdown.

---

## 6. Dataset CSV Schema & Output Format

The data collection tool (`ml/collect.py`) writes standardized records to `ml/datasets/collected_features.csv` with the following 12 columns:

```csv
person_id,exercise,camera_angle,knee_angle,hip_angle,torso_angle,knee_alignment,symmetry,movement_speed,rep_duration,label,fault_type
```

### Column Definitions
1. `person_id`: Anonymous identifier (`P001` to `P030`).
2. `exercise`: Target movement (e.g., `squat`, `push-up`, `walking-lunge`).
3. `camera_angle`: Angle of view (`front`, `45_degree`, `side`).
4. `knee_angle`: Lowest knee flexion angle reached during rep (degrees).
5. `hip_angle`: Hip angle at lowest point (degrees).
6. `torso_angle`: Torso inclination angle (degrees from horizontal).
7. `knee_alignment`: Knee valgus / varus deviation ratio.
8. `symmetry`: Left-vs-right knee/hip symmetry difference (degrees).
9. `movement_speed`: Mean angular velocity during transition (deg/s).
10. `rep_duration`: Total repetition duration in seconds.
11. `label`: Ground-truth risk classification (`LOW`, `MODERATE`, `HIGH`).
12. `fault_type`: Specific descriptive category (e.g. `none`, `knee_valgus`, `forward_lean`, `shallow_depth`, `excessive_speed`).

---

## 7. Operator Session Checklist

- [ ] Obtain signed Participant Consent Form (`ml/consent_form.md`).
- [ ] Assign new unique `person_id` (verify next available ID in sequence).
- [ ] Position camera and verify framing (head, knees, feet in frame during full range).
- [ ] Verify lighting and background clearance.
- [ ] Execute **Set 1: Correct Form** (10 reps) -> `label=LOW, fault_type=none`.
- [ ] 60-second rest period.
- [ ] Execute **Set 2: Minor Error** (10 reps) -> `label=MODERATE, fault_type=knee_valgus` (or chosen fault).
- [ ] 60-second rest period.
- [ ] Execute **Set 3: Dangerous Form** (10 reps, bodyweight only) -> `label=HIGH, fault_type=forward_lean` (or chosen fault).
- [ ] Verify CSV contains 30 valid rows for the participant.
- [ ] Store landmark archive in `ml/datasets/` and wipe temporary raw video.
