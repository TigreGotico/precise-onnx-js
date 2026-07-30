# MFCC Feature Extraction

Source: `src/mfcc.js`

This module is an exact port of `sonopy.mfcc_spec` (Python), which is the feature extractor used by `ovos-ww-plugin-precise-onnx`. All implementation decisions are validated against Python ground-truth vectors stored in `test/ww_vectors.json`.

---

## `mfccSpec(audio, sampleRate, windowSize, hopSize, numFilt, fftSize, numCoeffs)` → `Float32Array[]`

Computes MFCC features from a Float32 audio signal.

| Argument | Type | Default (Precise) | Description |
|----------|------|-------------------|-------------|
| `audio` | `Float32Array` | none | Normalized audio samples in `[-1, 1]` |
| `sampleRate` | number | `16000` | Sample rate in Hz |
| `windowSize` | number | `1600` | Analysis window size in samples (100 ms) |
| `hopSize` | number | `800` | Hop between windows in samples (50 ms) |
| `numFilt` | number | `20` | Number of mel filterbank filters |
| `fftSize` | number | `512` | FFT size (power of two) |
| `numCoeffs` | number | `13` | Number of MFCC coefficients to return |

**Returns:** array of `Float32Array`, one per frame. Each array has length `numCoeffs`.

Number of frames: `floor((audio.length - windowSize) / hopSize) + 1` (zero if `audio.length < windowSize`).

---

## Processing pipeline

### 1. Frame extraction (`chop_array`)

```
frames = [audio[i-windowSize : i] for i in range(windowSize, len(audio)+1, hopSize)]
```

No windowing function is applied. Frames are raw rectangular slices. This matches sonopy's `chop_array`, which uses no Hamming or Hann window.

### 2. Power spectrum (`powerSpec`)

For each frame, an FFT is computed and the power at each bin is:

```
power[k] = (re[k]² + im[k]²) / fftSize
```

Only the first `fftSize/2 + 1` bins are used (real FFT symmetry). The division by `fftSize` matches sonopy's convention.

The FFT is computed with an iterative in-place Cooley-Tukey algorithm (`_fftInPlace`).

### 3. Mel filterbank (`filterbanks`)

Triangular mel-scale filters are built using:

1. **Mel scale:** `1127 * ln(1 + f/700)`
2. **Grid:** `numFilt + 2` equally-spaced points on the mel scale from 0 Hz to `sampleRate` Hz
3. **Bin indices:** `grid_idx[i] = floor(melToHz(gridMels[i]) * nBins / sampleRate)`, with integer truncation
4. **`correct_grid`:** if consecutive grid indices are equal, the later one is pushed forward to ensure all filters are distinct
5. **Filter weights:** the rising edge is `linspace(0, 1, n, endpoint=False)` = `k/n`. The falling edge is `linspace(1, 0, n, endpoint=False)` = `(n-k)/n`.

The filterbank is cached per `(sampleRate, numFilt, nBins)` key (equivalent to Python's `@lru_cache`).

### 4. Log mel energies

```
logMel[m] = safeLog(dot(power, banks[m]))
```

where `safeLog(x) = ln(max(x, Number.EPSILON))`.

`Number.EPSILON === 2.220446049250313e-16 === np.finfo(float).eps`

### 5. DCT-II (`dctII`)

Ortho-normalized DCT-II matching `scipy.fft.dct(x, type=2, norm='ortho')`:

```
k=0:   y[0] = sum(x[n]) * sqrt(1/N)
k>0:   y[k] = sum(x[n] * cos(π·k·(2n+1)/(2N))) * sqrt(2/N)
```

### 6. First coefficient replacement

The `coeffs[0]` produced by the DCT is discarded and replaced with:

```
coeffs[0] = safeLog(sum(power))   // log-energy of the frame
```

This matches sonopy's final step. Coefficients `[1..numCoeffs-1]` are unchanged DCT output.

### 7. Float32 cast

Each frame is cast to `Float32Array` before being returned. This matches the precision of the ONNX model input and the Python reference (which stores frames in `np.float32`).

---

## Exported functions

| Function | Description |
|----------|-------------|
| `mfccSpec(...)` | Main entry point: full pipeline |
| `powerSpec(audio, windowSize, hopSize, fftSize)` | Frames → power spectra |
| `filterbanks(sampleRate, numFilt, nBins)` | Build (or cache) triangular mel filters |
| `dctII(x, numCoeffs)` | Ortho-normalized DCT-II |
| `safeLog(x)` | `ln(max(x, Number.EPSILON))` |

---

## Common mistakes (what NOT to do)

These are bugs found in naive JS ports of MFCC that break compatibility with sonopy:

| Bug | Correct behaviour |
|-----|-------------------|
| Apply Hamming/Hann window | No window: use rectangular frames |
| `power[k] = re[k]² + im[k]²` | Divide by `fftSize` |
| `log(max(x, 1e-6))` | Use `Number.EPSILON` (≈ 2.22e-16), not 1e-6 |
| `coeffs[0]` from DCT | Replace with `safeLog(sum(power))` |
| Use `nBins/sampleRate` Hz-to-bin formula without floor | `Math.trunc(hz * nBins / sampleRate)` |
| Skip `correct_grid` | Required when duplicate bin indices appear |

All of these were identified by comparing against Python vectors.

---
[Home](../README.md) · [Wake word API →](wakeword.md)
