# AI Usage Transparency

This document explains how and where AI assistance (Claude, by Anthropic) was used in the creation of `precise-onnx-js`.

---

## Summary

This library — including all source code, tests, test vector generators, and documentation — was written by Claude Sonnet 4.6 during a pair-programming session with a human developer. The human set direction, reviewed outputs, and approved the final results. Claude wrote essentially all of the code.

---

## What Claude did

### 1. Discovered bugs in the original MFCC implementation

The original `mfcc.js` in `hivemind-webspeech` had several bugs relative to `sonopy.mfcc_spec` (the Python reference used by `ovos-ww-plugin-precise-onnx`):

- Applied a Hamming window (sonopy uses no window — rectangular frames)
- Power formula missing `/fftSize` division
- Filterbank grid used a floating-point Hz-to-bin formula instead of `floor(hz * nBins / sr)`
- Missing `correct_grid` step (sonopy prevents duplicate filter indices)
- Wrong log epsilon (`1e-6` instead of `Number.EPSILON ≈ 2.22e-16`)
- First MFCC coefficient was the DCT output; sonopy replaces it with `log(sum(power))`

These bugs were found by Claude by inspecting the Python source via `inspect.getsource()` and comparing against the JS output.

### 2. Rewrote `mfcc.js` to exactly match sonopy

Claude rewrote `src/mfcc.js` from scratch by reading `sonopy`'s source code for `mfcc_spec`, `power_spec`, `filterbanks`, `correct_grid`, `chop_array`, and `safe_log`, then porting each function individually with inline comments explaining the equivalence.

### 3. Ported `wakeword.js` from Python

Claude ported `ThresholdDecoder`, `TriggerDetector`, and `PreciseOnnxWakeWord` from `ovos-ww-plugin-precise-onnx/inference.py`. This included:

- The PDF/CDF probability calibration logic in `ThresholdDecoder`
- The `correct_grid`-equivalent duplicate-prevention logic in filterbanks
- The rolling MFCC buffer and matrix-shift logic in `_updateVectors`
- A lazy `_getMfccSpec()` resolver for cross-environment compatibility (browser `<script>`, bundler, Node.js)

One JS-specific bug was fixed during this process: `-(x-mu)**2` is a SyntaxError in JS (unary minus before `**`); Claude rewrote it as `-(((x - mu) ** 2))`.

### 4. Created the entire `precise-onnx-js` repository

The human developer asked Claude to extract the wake word code into a standalone reusable library. Claude:

- Created the directory structure (`src/`, `test/`, `docs/`)
- Wrote `src/index.js` (CommonJS entry point with browser-global exports)
- Wrote `package.json` with correct `"type": "commonjs"`, `exports` map, and peer dependency declaration
- Updated `hivemind-webspeech` to depend on `precise-onnx-js` via a `file:` reference

### 5. Wrote the test suite (`test/wakeword.test.js`, 36 tests)

Claude designed and wrote the entire test suite, including:

- Choosing test cases for `ThresholdDecoder` (CDF shape, boundary inputs, 13 decode values)
- Designing the `TriggerDetector` sequences (4-consecutive-fire, cooldown, boundary at `prob = 0.5`)
- Writing the `_updateVectors` rolling-buffer tests and fixing two initially wrong test expectations (window drain size, float32 vs float64 first coefficient)
- Writing the MFCC equivalence tests against three audio types (zeros, sine, noise)
- Writing the mock-ONNX integration test

### 6. Wrote the Python vector generator (`test/generate_ww_vectors.py`)

Claude wrote the Python script that calls the reference Python libraries and writes `ww_vectors.json`, including installing `sonopy` and `onnxruntime` into the workspace venv to verify the script ran correctly.

### 7. Wrote all documentation

All files in `docs/` — including this one — were written by Claude.

---

## What the human did

- Set the overall direction: "create a precise-onnx-js repository and use that as a dependency, make the wakeword code reusable for others"
- Approved each step before Claude proceeded
- Approved (or implicitly accepted) the final code by not requesting changes
- Provided the workspace environment (Python venv, source repos, IDE)

---

## Why this matters

The core technical challenge in this library is that `sonopy.mfcc_spec` has several non-obvious implementation choices (no windowing, specific power normalization, `correct_grid`) that differ from textbook MFCC descriptions and from other JS MFCC libraries. Getting these wrong produces results that look plausible but fail on real models.

Claude found these discrepancies by reading both the Python source and the JS output, not by reading documentation. The validation approach (Python-generated vectors + JS tests comparing to 1e-3 tolerance) was also designed by Claude to make the correctness argument explicit and reproducible.

---

## Model

Claude Sonnet 4.6 (`claude-sonnet-4-6`), accessed via Claude Code CLI.
