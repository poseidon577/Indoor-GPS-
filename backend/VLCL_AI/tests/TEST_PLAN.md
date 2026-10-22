# Test plan by project step

Run everything from `backend/`: `PYTHONPATH=. python -m pytest -q` (about 40 s).
Expected result: all pass except one skip and the strict expected failures listed under step 3.

## Rules these tests follow

- **No mirrored implementation.** Expected values come from physics (the Lambertian model, Gaussian clipping
  statistics, geometry) or from independent checks, not from re-running the code under test.
- **Every fix has a "bug comes back" test.** The fix is disabled with a monkeypatch or flag, and the test
  asserts the old failure returns. This proves the passing test can actually fail.
- **Gaps are strict `xfail` specs.** Each one was confirmed to fail on today's code. When a gap is fixed,
  the test starts passing, strict mode fails the suite, and the marker must be removed.

## Step 1: understand and verify (tone grid fix)

`test_plan_step1_tone_grid.py`: off-grid detection; the snapped plan stays on the grid, keeps 5 distinct
bins and an arithmetic progression, and moves each tone by a bounded amount across 4 plans × 4 grids;
idempotence; sub-bin spacing; an aligned custom plan is left unchanged; cell error under 2 cm; **with
snapping disabled, error is over 10 cm**.

## Step 2: proposal Task 1

| File | Covers |
|---|---|
| `test_plan_step2_drive_gain.py` | Small signals untouched; exact gain into range for σ from 0.5 to 50; waveform shape kept; centre links error-free; corner links within FEC; **drive gain off → over 40 % clipping and broken BER**; **receiver ignoring the gain → bit errors** |
| `test_plan_step2_height_and_rss.py` | Solver uses the configured height, never the true height; a wrong height costs accuracy; calibration independent of receiver pose; RSS falls with distance; zero outside the FOV; noise unbiased and never negative; exact over the 25-point cell; 3 LEDs enough in 2D but not 3D; estimate bounded; error grows with noise, NLOS and tilt |
| `test_rss_localization.py` | Original RSS unit tests |
| `test_plan_step2_task1_experiment.py` | Feature/label column split; **no feature mirrors a label** (corr < 0.99); reproducible seeds; IMU tilt noisy, not exact; a pose that hides 2 LEDs does not poison the next pose (**mutation-checked**); feature builder; tilt-bin summary; MLP training sanity |

## Step 3: limitations (strict expected failures)

`test_plan_step3_known_gaps.py` and `../../test_pipeline.py`:

1. A-DPDOA never recovers after a pose that hides 2 LEDs (`prev_phases`).
2. A-DPDOA reports success while its estimate is pinned to the room corner.
3. A phone held at 1.0–1.1 m is off by about 1 m (2D mode assumes 0.85 m).
4. Localization does not degrade at 4 dB physics SNR, because the waveform ignores LED optical power.
5. The far LED fails to decode despite 58 dB physics SNR (same root cause).
6. NLOS is zero at most cell points, because the reflection walls assume a room spanning [0, W].
7. Reflected light carries no extra delay.
8. Legacy dataset features copy the true receiver x and y.
9. Legacy dataset features have no per-LED RSS.
10. The joint optimizer finds no feasible operating point.
11. `VLCLPipeline` (all modules end to end) cannot be constructed.

Not covered yet: whether the DL model generalises to other rooms and LED layouts. This needs a
multi-room dataset first.

## Step 4: novelty and paper

Nothing is testable yet. Navigation (proposal Task 2) and sonar (Task 3) have no design or code.
Their acceptance tests should be written once the design is agreed.
