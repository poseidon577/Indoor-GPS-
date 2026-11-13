# Task 1: localization benchmark (RSS vs A-DPDOA vs deep learning)

Proposal Task 1 asks for an RSS + Newton trilateration baseline and a deep-learning
localizer. This folder generates a clean dataset on the paper's 4-LED cell and
benchmarks both against A-DPDOA.

## Setup

- Room 5 x 5 x 3 m, LEDs at (±0.4, ±0.4, 1.35) m, 10 W each, receiver at 0.85 m.
- Receiver x, y uniform in [-0.5, 0.5] m; tilt uniform 0–20° in a random direction.
- NLOS reflections on; RSS gets 2 % multiplicative noise; IMU tilt gets 1° noise.
- A-DPDOA runs through the full integrated OFDM chain (Module 5) with clipping on.
- Features are only what a phone could measure (`feature_*`); ground truth stays in
  `label_*` / `meta_*`. 20k train, 2k val, 4k test poses, different seeds.
- MLP: 3 × 128 GELU, AdamW, smooth-L1, 300 epochs, best-val checkpoint, 3 seeds.

```
PYTHONPATH=. python experiments/task1_localization/generate_dataset.py --n 20000 --seed 1 --out experiments/task1_localization/data/train.csv
PYTHONPATH=. python experiments/task1_localization/generate_dataset.py --n 2000  --seed 2 --out experiments/task1_localization/data/val.csv
PYTHONPATH=. python experiments/task1_localization/generate_dataset.py --n 4000  --seed 3 --out experiments/task1_localization/data/test.csv
PYTHONPATH=. python experiments/task1_localization/train_and_benchmark.py --data experiments/task1_localization/data
```

## Results (test set, 2D error, cm; DL averaged over 3 seeds)

| Method | Mean | Median | P90 | Tilt 0–5° | Tilt 15–20° | LED out of FOV (20 % of poses) |
|---|---|---|---|---|---|---|
| RSS trilateration | 18.6 | 9.5 | 59.4 | 6.1 | 34.4 | 56.3 |
| A-DPDOA | 56.2 | 0.9 | 300.3 | 3.3 | 105.6 | 270.6 |
| DL: RSS | 3.9 | 3.1 | 7.9 | 2.0 | 6.0 | 4.7 |
| DL: RSS + IMU tilt | 1.4 | 1.1 | 2.3 | 1.1 | 1.9 | 2.3 |
| DL: RSS + phases | 1.1 | 0.3 | 3.1 | 0.3 | 2.3 | 4.1 |
| DL: RSS + phases + IMU tilt | 1.3 | 1.0 | 2.3 | 1.1 | 1.8 | 2.3 |

Seed-to-seed spread of the DL mean error is at most 0.03 cm. Full numbers are in
`data/results.json`.

## Caveats

- One room and one LED layout for train and test, so this measures interpolation,
  not generalisation to new rooms.
- The physics noise floor is very low (SNR 55–82 dB), so A-DPDOA phases are almost
  noiseless; its 1 cm accuracy with all LEDs visible is optimistic.
- A-DPDOA needs all four LEDs. When tilt pushes one outside the 70° field of view,
  the solve fails silently (it still reports success).
- The integrated engine tracks phases across steps (`prev_phases`). One bad step
  corrupts every later step, so the generator resets tracking for each pose.
