# Wake Word API Reference

Source: `src/wakeword.js`

Port of `ovos-ww-plugin-precise-onnx/inference.py`. Three classes: `ThresholdDecoder`, `TriggerDetector`, and `PreciseOnnxWakeWord`.

---

## `ThresholdDecoder`

Maps the raw sigmoid output of the ONNX model (a float in `[0, 1]`) to a calibrated probability. The mapping accounts for the distribution of the model's internal logit values.

### Constructor

```javascript
new ThresholdDecoder(muStds, center = 0.5, resolution = 200, minZ = -4, maxZ = 4)
```

| Parameter | Type | Precise default | Description |
|-----------|------|-----------------|-------------|
| `muStds` | `[[mu, std], ...]` | `[[6, 4]]` | Gaussian components of the logit distribution |
| `center` | number | `0.2` | CDF value that maps to probability 0.5 |
| `resolution` | number | `200` | Points per output unit in the CDF table |
| `minZ` | number | `-4` | Lower tail cutoff in standard deviations |
| `maxZ` | number | `4` | Upper tail cutoff in standard deviations |

With the Precise defaults, the CDF table has length `(maxOut - minOut) * resolution = 32 * 200 = 6400`.

### `decode(rawOutput)` → `number`

```javascript
const prob = decoder.decode(0.7); // → calibrated probability in [0, 1]
```

Steps:
1. Convert raw sigmoid output to logit space via `asigmoid(x) = -ln(1/x - 1)`
2. Clamp to `[minOut, maxOut]` and look up in the precomputed cumulative distribution `cd`
3. Linearly rescale: values below `center` map to `[0, 0.5]`, and values above map to `[0.5, 1]`

Special cases: `rawOutput === 0.0` returns `0`, and `rawOutput === 1.0` returns `1`.

### Static helpers

```javascript
ThresholdDecoder.sigmoid(x)   // 1 / (1 + exp(-x))
ThresholdDecoder.asigmoid(x)  // -ln(1/x - 1): inverse sigmoid
ThresholdDecoder.pdf(x, mu, std)  // Gaussian PDF
```

---

## `TriggerDetector`

Debounces the output of `ThresholdDecoder` so that rapid consecutive activations are counted but the detector only fires once per utterance.

### Constructor

```javascript
new TriggerDetector(chunkSize = 2048, sensitivity = 0.5, triggerLevel = 3)
```

| Parameter | Description |
|-----------|-------------|
| `chunkSize` | Audio chunk size in samples (used to compute cooldown duration) |
| `sensitivity` | Activation threshold: a chunk activates when `prob > 1 - sensitivity` |
| `triggerLevel` | Number of consecutive activations required before firing |

### `update(prob)` → `boolean`

Returns `true` when the wake word fires.

Internal `activation` counter behaviour:

- Activating chunk: `activation += 1`
- Non-activating chunk and `activation > 0`: `activation -= 1`
- `activation > triggerLevel`: fire! Reset `activation = -floor(8 * 2048 / chunkSize)`
- Negative `activation`: cooldown. The counter increments toward 0, which stops false triggers.

With defaults (`chunkSize=2048`, `triggerLevel=3`), cooldown is `-8` and the detector fires on the 4th consecutive high-probability chunk.

### Example

```javascript
const td = new TriggerDetector(2048, 0.5, 3);
td.update(0.9); // false: activation=1
td.update(0.9); // false: activation=2
td.update(0.9); // false: activation=3
td.update(0.9); // true: activation=-8 (cooldown)
td.update(0.9); // false: activation=-7 (still cooling down)
```

---

## `PreciseOnnxWakeWord`

Streaming wake word engine. Maintains a rolling MFCC buffer of the last 1.5 seconds of audio and runs the ONNX model on each new chunk.

### `static async PreciseOnnxWakeWord.load(modelUrl, threshold = 0.5, triggerLevel = 3)` → `PreciseOnnxWakeWord`

```javascript
const ww = await PreciseOnnxWakeWord.load('hey_mycroft.onnx');
```

Calls `ort.InferenceSession.create(modelUrl)`. Requires `globalThis.ort` to be set to `onnxruntime-web` (browser) or `onnxruntime-node` (Node.js).

### Constructor (direct use / testing)

```javascript
new PreciseOnnxWakeWord(session, inputName, threshold = 0.5, triggerLevel = 3)
```

Use `load()` in production. Direct construction is for testing with mock ONNX sessions.

### Model parameters (fixed)

| Property | Value | Description |
|----------|-------|-------------|
| `sampleRate` | 16000 | Hz |
| `windowSamples` | 1600 | 100 ms analysis window |
| `hopSamples` | 800 | 50 ms hop |
| `bufferSamples` | 24000 | 1.5 s rolling audio buffer |
| `nFeatures` | 29 | MFCC frames per inference |
| `nMfcc` | 13 | Coefficients per frame |
| `nFilt` | 20 | Mel filters |
| `nFft` | 512 | FFT size |

ONNX model input shape: `[1, 29, 13]` (batch × frames × coefficients).

### `async predict(float32Chunk)` → `boolean`

Feed a 2048-sample `Float32Array` from a 16 kHz microphone. Returns `true` when the wake word fires.

```javascript
const triggered = await ww.predict(chunk);
if (triggered) console.log('Wake word!');
```

Internally: `_updateVectors(chunk)` → `update()` → `triggerDetector.update(prob)`.

### `async update(float32Chunk)` → `number`

Returns the calibrated probability without debouncing. Useful for monitoring confidence.

```javascript
const prob = await ww.update(chunk); // 0.0 – 1.0
```

### `_updateVectors(chunk)` → `Float32Array`

Appends `chunk` to `windowAudio`, computes MFCC frames, rolls the `nFeatures × nMfcc` matrix, and drains consumed samples from `windowAudio`. Returns the flat MFCC matrix (row-major, `Float32Array` of length `nFeatures * nMfcc`).

### `clear()`

Resets `windowAudio` and the MFCC matrix to all-zeros. Call after a false positive or after a successful utterance to prevent trailing audio from influencing the next detection.

---

## Environment compatibility

| Environment | Setup |
|-------------|-------|
| Browser `<script>` | Load `mfcc.js` then `wakeword.js`, and load `ort` from a CDN before both |
| Browser bundler | `import { PreciseOnnxWakeWord } from 'precise-onnx-js'`. `ort` must be `globalThis.ort`. |
| Node.js | `globalThis.ort = require('onnxruntime-node')` before calling `load()` |

The `mfccSpec` function is resolved lazily at first call, via `_getMfccSpec()`. Load order of `mfcc.js` and `wakeword.js` does not matter, as long as both load before `predict()` is called.

---
[← MFCC feature extraction](mfcc.md) · [Home](../README.md) · [Testing →](testing.md)
