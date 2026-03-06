'use strict';

// precise-onnx-js — browser/Node.js port of ovos-ww-plugin-precise-onnx
//
// Browser via <script> tags (no bundler):
//   <script src="mfcc.js"></script>
//   <script src="wakeword.js"></script>
//   <!-- ThresholdDecoder, TriggerDetector, PreciseOnnxWakeWord, mfccSpec now global -->
//
// Browser via bundler (esbuild / webpack / vite):
//   import { PreciseOnnxWakeWord } from 'precise-onnx-js';
//   // or: const { PreciseOnnxWakeWord } = require('precise-onnx-js');
//
// Node.js:
//   const { PreciseOnnxWakeWord } = require('precise-onnx-js');
//   // ort (onnxruntime-node) must be set as globalThis.ort before calling load()

const mfccExports     = require('./mfcc.js');
const wakewordExports = require('./wakeword.js');

const lib = { ...mfccExports, ...wakewordExports };

// Expose as globals when running in a browser context (useful when loaded via <script>)
if (typeof globalThis !== 'undefined' && typeof window !== 'undefined') {
    Object.assign(globalThis, lib);
}

module.exports = lib;
