# Testing

## Running the test suite

```bash
# Run all 36 tests (no ONNX model required)
cd precise-onnx-js
node --test test/*.test.js
# or: npm test
```

Requires Node.js 18+. No `npm install` needed — the library has no runtime dependencies beyond an ONNX runtime (only needed for `load()`; tests mock the session).

## Regenerating Python reference vectors

```bash
# Requires sonopy and ovos-ww-plugin-precise-onnx in the Python venv
"/home/miro/PycharmProjects/HiveMind Workspace/.venv/bin/python" \
    test/generate_ww_vectors.py
```

Writes `test/ww_vectors.json`. Commit the updated file so JS test runs are reproducible without a Python environment.

---

## Test coverage (36 tests)

### `ThresholdDecoder` (6 tests)

| Test | What it checks |
|------|----------------|
| Constructor properties | `minOut`, `maxOut`, `outRange`, `center`, CDF length all match Python |
| CDF first 5 values | Floating-point match to 1e-12 tolerance |
| CDF last 5 values | Floating-point match to 1e-12 tolerance |
| `decode` on 13 values | Each decoded value matches Python to 1e-6 |
| Boundary: `decode(0)` → `0` | Exact |
| Boundary: `decode(1)` → `1` | Exact |

### `TriggerDetector` (7 tests)

| Test | What it checks |
|------|----------------|
| Seq 1 results match Python | Fire pattern: `[F,F,F,T,F,F,F,F,F,F,T]` — fires on 4th consecutive |
| Seq 1 activations match Python | Internal counter after each step |
| Seq 2 (all zeros) never fires | 10 steps of 0.0 → all false |
| Seq 3 (boundary 0.5) never fires | `prob > 1-sensitivity` is strict; 0.5 is not `> 0.5` |
| Cooldown value matches Python | `-floor(8 * 2048 / chunkSize) = -8` |
| Re-fires after cooldown | 8+ low-prob steps + 4 high → fires again |

### `PreciseOnnxWakeWord` properties (5 tests)

| Test | What it checks |
|------|----------------|
| `windowSamples` | 1600 |
| `hopSamples` | 800 |
| `bufferSamples` | 24000 |
| `nFeatures` | 29 |
| `nMfcc` | 13 |

All values compared against `precise_props` from `ww_vectors.json`.

### `_updateVectors` — rolling buffer (5 tests)

| Test | What it checks |
|------|----------------|
| Zero-length chunk returns zero matrix | No frames computed |
| Short chunk (< windowSamples) accumulates | No frames yet, audio stored |
| windowSamples chunk → 1 frame | Exactly one MFCC row produced |
| After consumption, buffer drains correctly | `windowAudio.length === hopSamples` after 1 frame |
| MFCC matrix rolls on second chunk | Oldest rows drop, new rows appended |

### `mfccSpec` equivalence — zeros (4 tests)

| Test | What it checks |
|------|----------------|
| Frame count matches Python | `n_frames` |
| `frame[0][0]` matches Python | Log-energy value (float32 precision) |
| `frame[0][1..12]` all zero | DCT of zero log-mels is zero |
| ATOL for all coefficients ≤ 1e-3 | All 13 coefficients, all frames |

### `mfccSpec` equivalence — 440 Hz sine (4 tests)

| Test | What it checks |
|------|----------------|
| Frame count matches Python | |
| `frame[0]` all coefficients within ATOL | 1e-3 |
| `frame[-1]` all coefficients within ATOL | Last frame |
| No NaN/Inf in any frame | Numerical stability check |

### `mfccSpec` equivalence — white noise seed 42 (4 tests)

| Test | What it checks |
|------|----------------|
| Frame count matches Python | |
| `frame[0]` all coefficients within ATOL | 1e-3 |
| `frame[-1]` all coefficients within ATOL | |
| No NaN/Inf | |

### Integration — predict pipeline (1 test)

| Test | What it checks |
|------|----------------|
| `predict` with mock session returning 0.99 | Fires after 4 consecutive calls; returns `boolean` |

---

## Cross-language validation methodology

1. `test/generate_ww_vectors.py` runs with the Python reference (`sonopy`, `ovos-ww-plugin-precise-onnx`) and writes exact float32 values to `test/ww_vectors.json`.
2. `test/wakeword.test.js` loads the JSON and compares JS output against those values.
3. MFCC coefficients use `ATOL = 1e-3` to account for float32 truncation (Python stores `np.float32`; intermediate calculations are float64 in both Python and JS).
4. Probability/CDF comparisons use `ATOL = 1e-6` or `1e-12` since those remain float64 throughout.

---

## Test vectors (`test/ww_vectors.json`)

| Key | Contents |
|-----|---------|
| `threshold_decoder` | `min_out`, `max_out`, `out_range`, `center`, `cd_len`, `cd_first5`, `cd_last5`, `decode` map |
| `trigger_detector` | `cooldown`, `seq1` (fire pattern + activations), `seq2`, `seq3` |
| `mfcc.zeros` | Frames for 4800-sample zero audio |
| `mfcc.sine_440` | Frames for 6400-sample 440 Hz sine |
| `mfcc.noise_seed42` | Frames for 6400-sample uniform noise (seed 42) |
| `precise_props` | `window_samples`, `hop_samples`, `buffer_samples`, `n_features`, `n_mfcc` |
