# Test plan by project step

Run everything from `backend/`: `PYTHONPATH=. python -m pytest -q` (about 6 min).
Expected result: all pass except one skip. Every step-3 gap below is closed; none is left as an expected failure.

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
| `test_plan_step2_height_and_rss.py` | (paper mode) Solver uses the configured height, never the true height; a wrong height costs accuracy; calibration independent of receiver pose; RSS falls with distance; zero outside the FOV; noise unbiased and never negative; exact over the 25-point cell; 3 LEDs enough in 2D but not 3D; estimate bounded; error grows with noise, NLOS and tilt |
| `test_rss_localization.py` | Original RSS unit tests |
| `test_plan_step2_task1_experiment.py` | Feature/label column split; **no feature mirrors a label** (corr < 0.99); reproducible seeds; IMU tilt noisy, not exact; a pose that hides 2 LEDs does not poison the next pose (**mutation-checked**); feature builder; tilt-bin summary; MLP training sanity |

## Step 3A: limitations found and fixed

Each gap was first a strict `xfail` spec, confirmed to fail, then fixed; the spec became a passing test.

| Gap | Fix | Tests |
|---|---|---|
| Waveform ignored LED optical power, so chain SNR was decoupled from physics; far LED failed to decode | LED power and combined photodiode noise carried through the chain | `test_plan_step3a1_signal_noise.py`: noise formula; error grows as lights dim; far LED within FEC; measured symbol noise = MMSE prediction within 1 dB |
| Reflections used the wrong room frame (zero at most points) and had no delay | Room frame inferred from the LEDs; reflections as a delayed tap | `test_plan_step3a2_reflections.py`: mirror symmetry (+ wrong-frame guard); gain vs independent brute-force integral within 5 %; patch convergence; reflected light arrives later; RSS bias in a ceiling room |
| A-DPDOA never recovered after a pose hiding 2 LEDs; reported success when pinned to a corner or when a hidden LED's tone came off a wall | Tone-SNR, boundary and residual checks with a status reason; phases tracked only after a VALID fix | `test_plan_step3a3_adpdoa_robustness.py` (mutation-checked): no false success over 150 random poses, no good pose rejected, recovery within 3 steps of every dropout on a 200-step walk |
| Phone height: paper's 2D solve is 4–8 cm off at 0.6–1.0 m even with all LEDs visible | Hybrid 3D solve: A-DPDOA distance differences + RSS read off the same tones + IMU tilt; flags the one ambiguous spot | `test_plan_step3a4_height.py`: tone RSS = LOS photocurrent; < 2 cm mean at 0.6–1.1 m and 0–15° tilt; no fix when an LED is hidden; height within 3 cm away from the ambiguous spot; the ambiguity is real (both roots fit exactly); guards: paper mode still off, IMU tilt needed |
| `VLCLPipeline` could not be built; adaptive mode lost the tones (short bursts, wrong tone plan); Module 6 used DC SNR (256-QAM everywhere, BER 5–19 %); allocation ignored; LED 1 got double the tone reserve; joint optimizer never feasible; legacy dataset leaked the true position and had no per-LED RSS | See the file's docstring for the 7 fixes | `test_plan_step3a5_plumbing.py`: pipeline localizes and meets BER in all 3 modes; one SNR model matches measurement within 1 dB; bit loading never exceeds supported order (+ guard); optimizer feasible at 3 poses, honestly infeasible in dim light; no position leak; per-LED RSS |

The 1 m errors first reported for a phone at 1.0 m came from poses with an LED out of view that were
not yet rejected; with 3A.3 those poses are reported as `TONE_NOT_DETECTED`, and the remaining height
error on visible poses is the 4–8 cm the hybrid solve removes.

Known, not yet addressed: Module 6's rate evaluator assumes a 20 MHz / 256 subcarrier width while the
grid runs at 50 MHz, so reported rates are a factor 2.5 low; generalisation to other rooms and layouts
needs the step-3B multi-room dataset.

## Step 4: novelty and paper

Nothing is testable yet. Navigation (proposal Task 2) and sonar (Task 3) have no design or code.
Their acceptance tests should be written once the design is agreed.
