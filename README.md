# precise-onnx-js

Browser/Node.js port of [ovos-ww-plugin-precise-onnx](https://github.com/OpenVoiceOS/ovos-ww-plugin-precise-onnx):
MFCC feature extraction + ONNX-based wake word detection compatible with
[Precise](https://github.com/MycroftAI/mycroft-precise) `.onnx` models.

## Live demo

Open [`index.html`](https://tigregotico.github.io/precise-onnx-js) in a browser (served over HTTPS or localhost). Select a wake word from the built-in list of models from [precise-lite-models](https://github.com/OpenVoiceOS/precise-lite-models), click **Load model**, then **Start microphone**.

<img width="752" height="837" alt="image" src="https://github.com/user-attachments/assets/4f75d7b5-288c-4cb7-8d5f-809529d17cfc" />

## Features

- **`mfccSpec`**: exact port of Python's `sonopy.mfcc_spec` (validated against Python test vectors)
- **`ThresholdDecoder`**: calibrates raw ONNX sigmoid output into a linear probability
- **`TriggerDetector`**: debounces consecutive activations
- **`PreciseOnnxWakeWord`**: streaming wake word engine with a rolling MFCC buffer

All classes are validated against the Python reference implementation using pytest-style vectors
stored in `test/ww_vectors.json`.

## Usage

### Browser (via `<script>` tags)

```html
<!-- Load onnxruntime-web first (exposes globalThis.ort) -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js"></script>
<!-- Load in order: mfcc before wakeword -->
<script src="precise-onnx-js/src/mfcc.js"></script>
<script src="precise-onnx-js/src/wakeword.js"></script>

<script>
async function main() {
    const ww = await PreciseOnnxWakeWord.load('hey_mycroft.onnx');
    // Feed 2048-sample Float32Array chunks from ScriptProcessorNode
    const triggered = await ww.predict(float32Chunk);
    if (triggered) console.log('Wake word detected!');
}
</script>
```

### Browser (via bundler: esbuild, webpack, or vite)

```js
import { PreciseOnnxWakeWord } from 'precise-onnx-js';
// ort must be available as globalThis.ort (import onnxruntime-web separately)
const ww = await PreciseOnnxWakeWord.load('hey_mycroft.onnx');
```

### Node.js (with onnxruntime-node)

```js
const ort = require('onnxruntime-node');
globalThis.ort = ort;

const { PreciseOnnxWakeWord } = require('precise-onnx-js');
const ww = await PreciseOnnxWakeWord.load('./hey_mycroft.onnx');
```

## API

### `mfccSpec(audio, sampleRate, windowSize, hopSize, numFilt, fftSize, numCoeffs)`

Computes MFCC features matching `sonopy.mfcc_spec`.

- `audio`: `Float32Array` of normalized audio samples
- Returns `Float32Array[]`: one row per frame, with length `numCoeffs`

Default parameters matching Precise models:
```
sampleRate=16000, windowSize=1600, hopSize=800, numFilt=20, fftSize=512, numCoeffs=13
```

### `class ThresholdDecoder(muStds, center, resolution, minZ, maxZ)`

Maps raw ONNX sigmoid output → calibrated probability.

- `muStds`: array of `[mu, std]` pairs (Precise default: `[[6, 4]]`)
- `center`: center point (Precise default: `0.2`)
- `.decode(rawOutput)` → `float` in `[0, 1]`

### `class TriggerDetector(chunkSize, sensitivity, triggerLevel)`

Prevents multiple rapid activations.

- `chunkSize=2048`, `sensitivity=0.5`, `triggerLevel=3`
- `.update(prob)` → `boolean`

### `class PreciseOnnxWakeWord`

#### `static async PreciseOnnxWakeWord.load(modelUrl, threshold=0.5, triggerLevel=3)`

Loads an ONNX model and returns a ready-to-use instance.

#### `async ww.predict(float32Chunk)` → `boolean`

Feed a 2048-sample chunk from a 16 kHz microphone. Returns `true` when the wake word fires.

#### `async ww.update(float32Chunk)` → `number`

Returns the calibrated detection probability without triggering the debouncer.

#### `ww.clear()`

Resets the rolling audio/MFCC buffer (e.g. after a false positive).

## Model parameters

The library is hardcoded to match Precise model defaults:

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| Window | 1600 samples (100 ms) |
| Hop | 800 samples (50 ms) |
| Buffer | 24000 samples (1.5 s) |
| MFCC features | 13 |
| Mel filters | 20 |
| FFT size | 512 |
| Model input shape | `[1, 29, 13]` |

## Running tests

```bash
# Generate Python reference vectors (requires sonopy, ovos-ww-plugin-precise-onnx)
python test/generate_ww_vectors.py

# Run JS tests (36 tests, no ONNX model required)
node --test test/*.test.js
```

See [`docs/testing.md`](docs/testing.md) for full test coverage details.

## Documentation

| File | Contents |
|------|---------|
| [`docs/mfcc.md`](docs/mfcc.md) | MFCC pipeline internals, sonopy equivalence, common porting mistakes |
| [`docs/wakeword.md`](docs/wakeword.md) | `ThresholdDecoder`, `TriggerDetector`, `PreciseOnnxWakeWord` API reference |
| [`docs/testing.md`](docs/testing.md) | Test suite details, vector generation, cross-language validation |
| [`docs/ai-usage.md`](docs/ai-usage.md) | AI usage transparency: how Claude was used to create this library |

## Relationship to hivemind-webspeech

[hivemind-webspeech](https://github.com/JarbasHiveMind/hivemind-webspeech) uses this library
as a dependency for its wake-word operating mode. `src/mfcc.js` and `src/wakeword.js` in that
repo are thin re-export wrappers. The canonical implementation lives here.

## Credits

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) / [NLnet](https://nlnet.nl)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429),
through the European Commission's [Next Generation Internet](https://ngi.eu) programme.
