# Experimental protocol

Wet-lab methodology for the larval-zebrafish TBI electrophysiology study that
this repository models.

> **Status: new integrated protocol; requires a pilot.**
> No published study has performed this exact combined experiment. Locskai et
> al. recorded acute behavior after TBI at 6 dpf and used a 3 dpf injury for a
> 7 dpf tau endpoint; Eimon et al. demonstrated penetrating forebrain LFP at
> 7 dpf; the high-speed pose study examined PTZ and genetic seizures at 3, 5,
> and 7 dpf. **3 dpf TBI followed by longitudinal 4–6 dpf LFP and behavior is a
> new integration of three published methods, not a replication of any one of
> them.** Every section below marks where the published work ends and the
> proposed integration begins.

---

## 1. Design

| Parameter | Specification |
|---|---|
| Fish | Wild-type AB larvae |
| Husbandry | 28 °C, 14 h light / 10 h dark, E3 medium |
| Injury | 3 dpf |
| Recording times | 24, 48, 72 h post-insult (4, 5, 6 dpf) |
| Groups | Sham, 100 g, 200 g, 300 g |
| Drop height | 108 cm (fixed) |
| Primary apparatus | 20 mL syringe in a three-prong clamp |
| Drops | One drop for the primary dose–response experiment |
| Behavior | Longitudinal, same larvae |
| LFP | Recoverable surface electrodes if the model needs the same fish across days; independent terminal cohorts if using the Eimon penetrating electrode exactly (see §6) |
| Biological replicates | ≥ 3 independent breeding clutches |
| Timing control | Every plate recorded at the same circadian time, within ± 30 min |

The 4–6 dpf window should initially be described as an **early post-traumatic
seizure or seizure-risk window**. It does not by itself establish chronic
post-traumatic epilepsy.

### Experimental unit

The **independent syringe/drop batch** is the experimental unit, with larvae
nested inside it. Larvae from one impact are nested observations, not
independent injury replicates. Clutch and insult batch must enter the
statistical model as grouping variables.

For a full-scale design, one plate per clutch gives 24 larvae per group; across
three clutches, 72 larvae per group. Split each condition into ≥ 2 independent
drop batches per clutch, giving six injury batches per condition rather than
three.

---

## 2. TBI drop apparatus

Based on the [Locskai TBI paper](https://doi.org/10.1242/bio.060601).

### Components

| Component | Specification |
|---|---|
| Syringe | 20 mL BD Luer-Lok, cat. 302830 |
| Medium in syringe | 1.0 mL E3 |
| Closure | Luer-Lok stopper valve, Cole-Parmer UZ-30600-00 |
| Holder | Three-prong clamp, centered on a support rod and stand |
| Guide tube | Vertical, 108 cm |
| Guide-tube diameter | ≈ 40 mm for a 38 mm calibration weight; match the tube closely to the actual weight |
| Drop masses | 100, 200, 300 g calibrated cylindrical weights |
| Sham | Identical loading, waiting, transfer, and handling; no dropped weight |
| Air | No visible bubbles before impact |
| Alignment | Weight centerline, guide tube, and plunger center coaxial |
| Larvae per batch | Identical across groups; 12–24 per independent drop batch is reasonable for a pilot |

**Safety:** splash shield, safety glasses, secured guide tube, and a mechanical
stop preventing the weight from leaving the apparatus.

### Procedure

1. Place a fixed number of 3 dpf larvae in 1.0 mL E3 in the syringe.
2. Close the Luer valve.
3. Inspect for air bubbles; reopen and reload if any are present.
4. Secure the syringe vertically in the calibration clamp position.
5. Set the plunger to the predefined starting position.
6. Center the 108 cm guide tube over the plunger.
7. **Release** — do not push — the calibrated weight from the top of the guide.
8. Record the pressure trace and the exact injury timestamp.
9. Open the valve and transfer larvae immediately to assigned wells.
10. Record heartbeat, survival, posture, and touch/fin-poke response every
    5 min for 30 min.

Repeated impacts (Locskai: bubbles removed and larvae reset between impacts,
< 10 min apart) must be a **separate experiment**: drop count, pressure, and
mortality become confounded otherwise.

### Pressure calibration

**Do not treat "100 g at 108 cm" as the biological dose.** The dose is the
measured pressure waveform.

Published acquisition system:

- Arduino Uno Rev3
- AUTEX GSND-0556629788 pressure transducer
- Photoresistor at the top of the guide tube to trigger acquisition
- 1 s pressure acquisition
- Arduino IDE 2.0.3 at 2,000,000 baud
- Static-weight calibration using Pascal's principle

Conversion:

```text
voltage = (5 × output) / 1023
PSI     = (voltage − 0.5) × 37.5
kPa     = PSI × 6.895
```

Retain for every calibration trace: maximum pressure (kPa), mean pressure of
the primary wave, primary-wave duration, number of rebound waves, time between
first and second waves, syringe lot, plunger starting position, holder, height,
mass, and drop number.

Run ≥ 3 E3-only calibration drops per setting before animal work, and bracket
animal batches with repeat calibration.

> **Interpretation limit.** Locskai found behavioral seizure phenotypes at
> roughly 90–300 kPa, while pressures above ≈ 300 kPa could suppress gross
> locomotion. **Low movement therefore cannot be read as absence of seizures.**

The 10 mL syringe plus rigid foam block produced the broadest range
(≈ 33–1105 kPa) but is not recommended as the initial behavioral configuration:
it risks severe injury, inactivity, and mortality. The 20 mL clamp setup matches
the published seizure dose–response experiment and the 100/200/300 g design.

---

## 3. Plate and well design

The high-speed behavior study used one larva per well in 150 µL, but did not
report plate manufacturer or bottom geometry. **Those values must be recorded
from the plate actually purchased, not assumed.**

| Parameter | Specification |
|---|---|
| Format | 96-well, 8 × 12 |
| Bottom | Clear, optically flat, flat-bottom |
| Well shape | Round |
| Density | One larva per well |
| Volume | 150 µL E3 |
| Habituation | 20 min before high-speed recording |
| Temperature | 28 ± 0.5 °C |
| Lid | Clear, low-condensation |
| Identity | Same fish ID and well ID from 3 to 6 dpf |
| Calibration | Record measured well diameter, medium depth, and pixels/mm |

The Whyte-Fagundes study referenced a circular well area of 0.32 cm², implying
≈ 6.38 mm diameter. **That diameter is derived, not reported** — measure the
purchased plate.

### Layout

Interleave conditions; never confine a group to one row or plate region.

```text
A/E: Sham  100g  200g  300g  Sham  100g  200g  300g  Sham  100g  200g  300g
B/F: 100g  200g  300g  Sham  100g  200g  300g  Sham  100g  200g  300g  Sham
C/G: 200g  300g  Sham  100g  200g  300g  Sham  100g  200g  300g  Sham  100g
D/H: 300g  Sham  100g  200g  300g  Sham  100g  200g  300g  Sham  100g  200g
```

Randomize larvae into assigned wells.

---

## 4. High-speed imaging

Published parameters from
[Whyte-Fagundes et al.](https://doi.org/10.1038/s42003-025-08310-6):

| Parameter | Value |
|---|---|
| Platform | Ramona Optics MCAM |
| Camera array | 24 high-resolution cameras |
| Whole-plate resolution | 312 megapixels |
| Frame rate | 160 fps |
| Exposure | 2 ms |
| Digital gain | 2.0 |
| Analog gain | 1.25 |
| Illumination | 850 nm transmitted infrared, 65 % brightness |
| Sensor binning | Mode 4 |
| Per-well output | 256 × 256 px |
| Epoch duration | 5 min |
| Frames per well/epoch | 48,000 |
| Raw format | NetCDF with metadata |
| Compressed format | MP4 |
| Focus | Platform height adjusted before each experiment |

Image a calibration target at the well plane at the start of every plate
session and store the pixels/mm value with the video metadata.

### The first-seizure timing problem

**Five-minute epochs cannot establish the first seizure between 24 and 72 h.**
They establish only the first seizure *observed during a sampled epoch*.

For a defensible `first_seizure_hours`, use two recording layers:

1. **Continuous surveillance** of the whole plate at ≈ 25–30 fps under 850 nm.
2. **Scheduled or triggered 160 fps MCAM** recordings for pose and seizure
   classification.

Without continuous surveillance, report **`first_observed_seizure_hours`** and
preserve the previous and next observation times as **interval-censoring
bounds**. Do not present a sampled detection time as an exact event time.

---

## 5. Pose estimation and behavioral classification

The Whyte-Fagundes pose model used a DeepLabCut backbone subsequently optimized
for MCAM. A standard DeepLabCut implementation is therefore an **adaptation, not
an exact reproduction**.

### Keypoints

Snout · left eye · right eye · body center · tail 1 · tail 2 · tail 3 ·
caudal-fin/tail tip.

Output `x`, `y`, and likelihood for every keypoint and frame.

Train on ≥ 1,000 diverse TBI-specific frames spanning sham and all three injury
levels; 4, 5, and 6 dpf; all clutches and plate positions; straight swimming,
edge swimming, whirlpooling, convulsion, posture loss, blur, and occlusion; and
both low- and high-activity larvae.

Per the [DeepLabCut protocol](https://doi.org/10.1038/s41596-019-0176-0), use
held-out evaluation and iterative refinement, and **split by fish, video, and
clutch — not by randomly mixed frames from the same videos.**

### Pose QC

Remove frames when calculated speed exceeds 120 mm/s, the body-center keypoint
leaves the well boundary, or any keypoint is more than 0.7 body lengths from the
center of mass. Then apply sym4 wavelet denoising.

Extract instantaneous speed, distance traveled, heading-angle change,
tail-angle change, inter-eye distance, time near the well wall, pose likelihood,
and missing-frame fraction.

> **120 mm/s is an artifact-removal threshold, not a seizure threshold.**
> Likewise, a rule such as "speed above 20 mm/s" does not by itself identify a
> seizure.

### Behavioral classes

Stationary · normal swim · whirlpool · convulsion · posture loss.

The published classifier used overlapping 60-frame windows (0.375 s at 160 fps),
labeled 4,418 clips, and selected a random forest over k-NN and SVM
alternatives.

A qualifying seizure must be a confirmed Baraban/Locskai **Stage II or III**
event:

| Stage | Description |
|---|---|
| I | Increased activity or brief darting |
| II | Rapid whirlpool / circular swimming |
| III | Clonus-like whole-body convulsion followed by loss of posture |

Locskai operationalized Stage II/III as whirlpool movement, clonic events, or
being fully on the side with inactivity longer than one second. See the
[Locskai paper](https://doi.org/10.1242/bio.060601) and the original
[Baraban seizure model](https://doi.org/10.1016/j.neuroscience.2004.11.031).

### Outcome definitions

```text
became_epileptic = 1
    if at least one blinded-confirmed spontaneous Stage II/III event
    occurs during the prespecified 24-72 h observation window

became_epileptic = 0
    if the complete required observation window is available
    and no qualifying event occurs

first_seizure_hours =
    (timestamp of first qualifying Stage II/III event
     - exact TBI timestamp) / 3600
```

> **Dead larvae, lost fish, and larvae with inadequate video coverage are `NA`,
> never `0`.** An unobserved animal has an unknown outcome, not a negative one;
> coding absence as negative inflates the negative class and biases every
> downstream rate. This rule is enforced in code — see §7.

Because one seizure is not chronic epilepsy, describe this column as an
**operational early post-traumatic seizure outcome**. A stricter secondary
definition may require recurrent events.

### Scoring reliability

≥ 20 % of clips independently scored by two people blinded to TBI group. Report
Cohen's κ between human scorers; Cohen's κ between automated and consensus
scoring; per-class precision, recall, F1, and confusion matrix; pose error in
micrometers; and performance separately at 4, 5, and 6 dpf.

---

## 6. LFP

### Compatibility constraint (decide before collecting data)

Eimon's method inserts an electrode **into the forebrain**. It was demonstrated
at 7 dpf and **was not validated as a recoverable, repeated measurement in the
same larva at 4, 5, and 6 dpf.**

| If the analysis requires… | Use |
|---|---|
| The same fish across all three days, with final per-fish behavior | A **recoverable surface-electrode** system. The [Hong iZAP system](https://doi.org/10.1038/srep28248) recorded non-invasively from 3–7 dpf larvae and allowed fish retrieval. |
| The Eimon penetrating electrode exactly | **Independent terminal cohorts** at 4, 5, and 6 dpf — and **do not claim individual longitudinal state transitions.** |

This is a hard constraint on the modeling, not a footnote: a Markov model over
per-fish daily transitions presupposes the first column.

### Eimon apparatus

From [Eimon et al.](https://doi.org/10.1038/s41467-017-02404-4):

- Glass animal capillary: 1.1 mm ID, 1.5 mm OD
- Recording electrode pulled from 1 mm OD capillary
- Electrode filled with 1 M chloride solution
- Ag/AgCl wire
- RHD2216 16-channel Intan preamplifier; RHD2000 acquisition system
- Forebrain placement
- Stop advancement at ≈ 3 MΩ, noise below 0.2 mV RMS
- Exclude recordings if resistance changes by more than 50 %
- No paralytic
- Inner agarose core 1.3 % ultra-low-gelling; outer shell 2 % low-gelling

### Acquisition and preprocessing

- First-order 3 kHz low-pass anti-alias filter
- Sampling at 3 kSamples/s
- Offline second-order Butterworth band-pass, 0.5–1000 Hz
- 30 s windows, 20 s overlap
- ≥ 30 min spontaneous recording per session

> **Blinding.** The unsupervised Markov model receives LFP-derived features
> only. Behavioral outcomes stay sealed until model fitting, state-number
> selection, preprocessing, and risk-score calculation are frozen.

---

## 7. Synchronization and required metadata

Every record carries:

```text
fish_id                 clutch_id               insult_batch_id
plate_id                well_id                 insult_timestamp
recording_start_timestamp                       hours_post_insult
drop_mass_g             drop_height_cm          syringe_volume_ml
holder_type             drop_count              peak_pressure_kpa
primary_wave_mean_kpa   pressure_wave_count     video_id
lfp_recording_id        camera_fps              pixels_per_mm
pose_model_version
```

Compute `hours_post_insult` from timestamps; never type it manually. Use one
synchronized computer clock for pressure, camera, and LFP acquisition.

Include **clutch** and **insult batch** as grouping variables in every
statistical model.

---

## 8. How the dataset maps to this protocol

The dataset in `data/measured/` matches the apparatus specification in §1–2:
20 mL syringe, three-prong clamp, 108 cm, **single drop**, 100/200/300 g, with
measured peak pressures of 115 / 210 / 319 kPa — inside and at the top of the
90–300 kPa behavioral-seizure range.

| Protocol element | Dataset status |
|---|---|
| 3 dpf injury, 4/5/6 dpf recording | Present |
| Sham + 3 doses, 60 fish each | Present |
| Clutch and batch identifiers | Present (6 clutches, 3 batches) |
| Measured peak pressure | Present, per fish |
| Blinded Baraban staging | Present, `reviewer_blinded` true throughout |
| Pose QC flags | Present |
| **`insult_batch_id`** | **Absent** — drop batch cannot be modeled as the experimental unit |
| **`first_seizure_hours`** | **Empty (240/240)**; reconstructed from sampled sessions, so interval-censored |
| **Continuous surveillance layer** | **Absent** — see §4 |

### Three gaps that bound the current claims

1. **The electrode metadata indicates the penetrating preparation.**
   `electrode_target = forebrain`, `electrode_solution = 1 M chloride`, and
   impedance 2.45–3.57 MΩ match Eimon's ≈ 3 MΩ stop criterion — yet 228 fish
   have all three daily sessions. Under §6 that combination is the case
   requiring terminal cohorts, in which **individual longitudinal state
   transitions should not be claimed.** The Markov model assumes exactly those
   transitions, so this is an assumption the dataset cannot verify. It is the
   single most important thing a pilot must settle.

2. **The seizure endpoint is interval-censored, not exact.** Behavior is scored
   in three discrete sessions, so the reconstructed timing is
   `first_observed_seizure_hours` in the sense of §4. The endpoint therefore
   uses **presence of a qualifying event in the 6 dpf session**, not a
   continuous latency.

3. **Seven fish have no 6 dpf observation.** Per §5 they are coded `NA`, not
   `0`. They are excluded from endpoint scoring rather than counted as
   negatives. This is enforced in `tbi_markov.dataset._dpf6_endpoint` and
   covered by `tests/test_dataset.py`.

---

## References

1. Locskai LF, Gill T, Tan SAW, et al. *A larval zebrafish model of traumatic
   brain injury.* Biology Open. 2025;14(2):bio060601.
   [doi:10.1242/bio.060601](https://doi.org/10.1242/bio.060601)
2. Eimon PM, Ghannad-Rezaie M, De Rienzo G, et al. *Brain activity patterns in
   high-throughput electrophysiology screen predict both drug efficacies and
   side effects.* Nature Communications. 2018;9:219.
   [doi:10.1038/s41467-017-02404-4](https://doi.org/10.1038/s41467-017-02404-4)
3. Whyte-Fagundes P, et al. *High-speed pose estimation of larval zebrafish
   seizure behavior.* Communications Biology. 2025.
   [doi:10.1038/s42003-025-08310-6](https://doi.org/10.1038/s42003-025-08310-6)
4. Hong S, et al. *iZAP: non-invasive zebrafish electrophysiology.* Scientific
   Reports. 2016;6:28248. [doi:10.1038/srep28248](https://doi.org/10.1038/srep28248)
5. Nath T, Mathis A, Chen AC, et al. *Using DeepLabCut for 3D markerless pose
   estimation.* Nature Protocols. 2019;14:2152–2176.
   [doi:10.1038/s41596-019-0176-0](https://doi.org/10.1038/s41596-019-0176-0)
6. Baraban SC, Taylor MR, Castro PA, Baier H. *Pentylenetetrazole induced
   changes in zebrafish behavior, neural activity and c-fos expression.*
   Neuroscience. 2005;131(3):759–768.
   [doi:10.1016/j.neuroscience.2004.11.031](https://doi.org/10.1016/j.neuroscience.2004.11.031)
