// Wake word detection — port of ovos-ww-plugin-precise-onnx/inference.py
//
// Works in three environments:
//   Browser via <script>: mfcc.js must be loaded first (sets globalThis.mfccSpec)
//   Browser via bundler: import/require resolves mfcc.js automatically
//   Node.js: require('./mfcc.js') used automatically

'use strict';

// ── mfccSpec resolver ─────────────────────────────────────────────────────────
// Lazy: resolved on first _updateVectors call so load order doesn't matter.

function _getMfccSpec() {
    if (typeof globalThis !== 'undefined' && typeof globalThis.mfccSpec === 'function') {
        return globalThis.mfccSpec;
    }
    if (typeof require !== 'undefined') {
        return require('./mfcc.js').mfccSpec;
    }
    throw new Error('precise-onnx-js: mfccSpec not found. ' +
                    'Load mfcc.js before wakeword.js or use a bundler.');
}

// ── ThresholdDecoder ──────────────────────────────────────────────────────────
// Maps raw ONNX sigmoid output → calibrated probability in [0, 1].
// Port of Python's inference.ThresholdDecoder.

class ThresholdDecoder {
    constructor(muStds, center = 0.5, resolution = 200, minZ = -4, maxZ = 4) {
        this.minOut = Math.trunc(Math.min(...muStds.map(([mu, std]) => mu + minZ * std)));
        this.maxOut = Math.trunc(Math.max(...muStds.map(([mu, std]) => mu + maxZ * std)));
        this.outRange = this.maxOut - this.minOut;
        this.center = center;
        this.cd = this._cumsum(this._calcPd(muStds, resolution));
    }

    static sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
    static asigmoid(x) { return -Math.log(1 / x - 1); }

    static pdf(x, mu, std) {
        if (std === 0) return 0;
        return (1 / (std * Math.sqrt(2 * Math.PI))) *
               Math.exp(-((x - mu) ** 2) / (2 * std ** 2));
    }

    _linspace(start, end, n) {
        const arr = new Float64Array(n);
        const step = (end - start) / (n - 1);
        for (let i = 0; i < n; i++) arr[i] = start + i * step;
        return arr;
    }

    _cumsum(arr) {
        const out = new Float64Array(arr.length);
        let s = 0;
        for (let i = 0; i < arr.length; i++) { s += arr[i]; out[i] = s; }
        return out;
    }

    _calcPd(muStds, resolution) {
        const n = resolution * this.outRange;
        if (n === 0) return new Float64Array(0);
        const points = this._linspace(this.minOut, this.maxOut, n);
        const pd = new Float64Array(n);
        const norm = resolution * muStds.length;
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (const [mu, std] of muStds) sum += ThresholdDecoder.pdf(points[i], mu, std);
            pd[i] = sum / norm;
        }
        return pd;
    }

    decode(rawOutput) {
        if (rawOutput === 1.0 || rawOutput === 0.0) return rawOutput;
        let cp;
        if (this.outRange === 0) {
            cp = rawOutput > this.minOut ? 1 : 0;
        } else {
            let ratio = (ThresholdDecoder.asigmoid(rawOutput) - this.minOut) / this.outRange;
            ratio = Math.min(1, Math.max(0, ratio));
            cp = this.cd[Math.round(ratio * (this.cd.length - 1))];
        }
        if (cp < this.center) {
            return 0.5 * cp / this.center;
        } else {
            return 0.5 + 0.5 * (cp - this.center) / (1 - this.center);
        }
    }
}

// ── TriggerDetector ───────────────────────────────────────────────────────────
// Debounces consecutive activations. Port of Python's inference.TriggerDetector.

class TriggerDetector {
    constructor(chunkSize, sensitivity = 0.5, triggerLevel = 3) {
        this.chunkSize    = chunkSize;
        this.sensitivity  = sensitivity;
        this.triggerLevel = triggerLevel;
        this.activation   = 0;
    }

    update(prob) {
        const chunkActivated = prob > 1.0 - this.sensitivity;
        if (chunkActivated || this.activation < 0) {
            this.activation += 1;
            const hasActivated = this.activation > this.triggerLevel;
            if (hasActivated || (chunkActivated && this.activation < 0)) {
                this.activation = -Math.floor(8 * 2048 / this.chunkSize);
            }
            if (hasActivated) return true;
        } else if (this.activation > 0) {
            this.activation -= 1;
        }
        return false;
    }
}

// ── PreciseOnnxWakeWord ───────────────────────────────────────────────────────
// Streaming MFCC + ONNX inference. Port of Python's PreciseOnnxEngine.
//
// Usage (browser):
//   const ww = await PreciseOnnxWakeWord.load('hey_mycroft.onnx');
//   const triggered = await ww.predict(float32ChunkOf2048);
//
// Usage (Node.js test with mock session):
//   const ww = new PreciseOnnxWakeWord({ inputNames: ['input'] }, 'input');
//   ww._updateVectors(audioChunk);

class PreciseOnnxWakeWord {
    constructor(session, inputName, threshold = 0.5, triggerLevel = 3) {
        this.sampleRate = 16000;
        this.nMfcc      = 13;
        this.nFilt      = 20;
        this.nFft       = 512;
        this.hopT       = 0.05;
        this.windowT    = 0.1;
        this.bufferT    = 1.5;

        this.thresholdDecoder = new ThresholdDecoder([[6, 4]], 0.2);
        this.triggerDetector  = new TriggerDetector(2048, threshold, triggerLevel);

        this.session   = session;
        this.inputName = inputName;

        this.windowAudio = new Float32Array(0);
        // Flat row-major (nFeatures × nMfcc), all zeros
        this.mfccs = new Float32Array(this.nFeatures * this.nMfcc);
    }

    get windowSamples() { return Math.round(this.sampleRate * this.windowT); }  // 1600
    get hopSamples()    { return Math.round(this.sampleRate * this.hopT); }     //  800
    get bufferSamples() {
        const raw = Math.round(this.sampleRate * this.bufferT);                 // 24000
        return this.hopSamples * Math.floor(raw / this.hopSamples);             // 24000
    }
    get nFeatures() {
        // 1 + floor((24000 - 1600) / 800) = 29
        return 1 + Math.floor((this.bufferSamples - this.windowSamples) / this.hopSamples);
    }

    // Factory — loads ONNX model from URL (browser) or file path (Node.js).
    // Requires globalThis.ort (onnxruntime-web or onnxruntime-node) to be loaded.
    static async load(modelUrl, threshold = 0.5, triggerLevel = 3) {
        const session   = await ort.InferenceSession.create(modelUrl);
        const inputName = session.inputNames[0];
        return new PreciseOnnxWakeWord(session, inputName, threshold, triggerLevel);
    }

    clear() {
        this.windowAudio = new Float32Array(0);
        this.mfccs = new Float32Array(this.nFeatures * this.nMfcc);
    }

    // Append chunk to rolling audio window, compute MFCCs, update mfccs matrix.
    // Returns flat mfccs Float32Array (nFeatures × nMfcc).
    _updateVectors(chunk) {
        const combined = new Float32Array(this.windowAudio.length + chunk.length);
        combined.set(this.windowAudio);
        combined.set(chunk, this.windowAudio.length);
        this.windowAudio = combined;

        if (this.windowAudio.length >= this.windowSamples) {
            const mfccSpec = _getMfccSpec();
            const frames = mfccSpec(
                this.windowAudio,
                this.sampleRate,
                this.windowSamples,
                this.hopSamples,
                this.nFilt,
                this.nFft,
                this.nMfcc
            ); // Float32Array[], each length nMfcc

            // Advance window past consumed samples
            this.windowAudio = this.windowAudio.slice(frames.length * this.hopSamples);

            // Clamp to nFeatures
            const incoming = frames.length > this.nFeatures
                ? frames.slice(frames.length - this.nFeatures)
                : frames;

            // Roll matrix: drop oldest `incoming.length` rows, append new ones
            const nKeep = this.nFeatures - incoming.length;
            const next  = new Float32Array(this.nFeatures * this.nMfcc);
            if (nKeep > 0) {
                next.set(this.mfccs.subarray(incoming.length * this.nMfcc), 0);
            }
            for (let i = 0; i < incoming.length; i++) {
                next.set(incoming[i], (nKeep + i) * this.nMfcc);
            }
            this.mfccs = next;
        }

        return this.mfccs;
    }

    // Process one audio chunk. Returns calibrated probability (0–1).
    async update(chunk) {
        const mfccs  = this._updateVectors(chunk);
        const tensor = new ort.Tensor('float32', mfccs, [1, this.nFeatures, this.nMfcc]);
        const results = await this.session.run({ [this.inputName]: tensor });
        const rawOutput = Object.values(results)[0].data[0];
        return this.thresholdDecoder.decode(rawOutput);
    }

    // Returns true if this chunk triggers the wake word.
    async predict(chunk) {
        const prob = await this.update(chunk);
        return this.triggerDetector.update(prob);
    }
}

if (typeof module !== 'undefined') {
    module.exports = { ThresholdDecoder, TriggerDetector, PreciseOnnxWakeWord };
}
