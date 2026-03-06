"""
Generate ww_vectors.json — ground-truth values from the Python reference
implementation (ovos-ww-plugin-precise-onnx / sonopy).

Run with:
  "/home/miro/PycharmProjects/HiveMind Workspace/.venv/bin/python" \
      hivemind-webspeech/test/generate_ww_vectors.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/miro/PycharmProjects/OpenVoiceOS Workspace/ovos-ww-plugin-precise-onnx')
from ovos_ww_plugin_precise_onnx.inference import ThresholdDecoder, TriggerDetector
from sonopy import mfcc_spec

OUT = os.path.join(os.path.dirname(__file__), 'ww_vectors.json')

vectors = {}

# ── ThresholdDecoder ──────────────────────────────────────────────────────────
td = ThresholdDecoder([(6, 4)], center=0.2)

test_inputs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
decode_map = {}
for x in test_inputs:
    decode_map[str(x)] = td.decode(float(x))

vectors['threshold_decoder'] = {
    'mu_stds':   [[6, 4]],
    'center':    0.2,
    'min_out':   int(td.min_out),
    'max_out':   int(td.max_out),
    'out_range': int(td.out_range),
    'cd_len':    int(len(td.cd)),
    'cd_first5': td.cd[:5].tolist(),
    'cd_last5':  td.cd[-5:].tolist(),
    'decode':    decode_map,
}

# ── TriggerDetector ───────────────────────────────────────────────────────────

# Sequence 1: 4 consecutive high → fires on 4th, then inhibits
trig1 = TriggerDetector(chunk_size=2048, sensitivity=0.5, trigger_level=3)
seq1 = [0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9]
res1 = []
act1 = []
for p in seq1:
    res1.append(bool(trig1.update(float(p))))
    act1.append(int(trig1.activation))

# Sequence 2: all zeros → never fires
trig2 = TriggerDetector(chunk_size=2048, sensitivity=0.5, trigger_level=3)
seq2 = [0.0] * 10
res2 = [bool(trig2.update(float(p))) for p in seq2]

# Sequence 3: exactly at threshold boundary (prob = 0.5 == 1.0 - sensitivity)
trig3 = TriggerDetector(chunk_size=2048, sensitivity=0.5, trigger_level=3)
seq3 = [0.5, 0.5, 0.5, 0.5, 0.5]
res3 = [bool(trig3.update(float(p))) for p in seq3]

# Cooldown: after firing, how many steps of low prob before can fire again?
cooldown = -(8 * 2048) // 2048   # Python: -(8*2048)//chunk_size

vectors['trigger_detector'] = {
    'chunk_size':   2048,
    'sensitivity':  0.5,
    'trigger_level': 3,
    'cooldown':     int(cooldown),
    'seq1': {'inputs': seq1, 'results': res1, 'activations_after': act1},
    'seq2': {'inputs': seq2, 'results': res2},
    'seq3': {'inputs': seq3, 'results': res3},
}

# ── MFCC (mfccSpec) ───────────────────────────────────────────────────────────
SR       = 16000
WIN      = 1600
HOP      = 800
N_FILT   = 20
N_FFT    = 512
N_MFCC   = 13

def compute_mfccs(audio):
    frames = mfcc_spec(audio, SR, (WIN, HOP), num_filt=N_FILT, fft_size=N_FFT, num_coeffs=N_MFCC)
    return frames

# Test 1: zeros  (log(0 + 1e-6) floor → predictable DCT)
zeros = np.zeros(WIN * 3, dtype=np.float32)
f_zeros = compute_mfccs(zeros)

# Test 2: 440 Hz sine wave
t = np.arange(WIN * 4, dtype=np.float32) / SR
sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
f_sine = compute_mfccs(sine)

# Test 3: white noise with fixed seed for reproducibility
rng = np.random.default_rng(42)
noise = rng.uniform(-1.0, 1.0, WIN * 4).astype(np.float32)
f_noise = compute_mfccs(noise)

vectors['mfcc'] = {
    'params': {'sample_rate': SR, 'window_samples': WIN, 'hop_samples': HOP,
               'n_filt': N_FILT, 'n_fft': N_FFT, 'n_mfcc': N_MFCC},
    'zeros': {
        'audio_len': int(len(zeros)),
        'n_frames':  int(len(f_zeros)),
        'frames':    [row.tolist() for row in f_zeros],
    },
    'sine_440': {
        'audio_len': int(len(sine)),
        'n_frames':  int(len(f_sine)),
        'frames':    [row.tolist() for row in f_sine],
    },
    'noise_seed42': {
        'audio_len': int(len(noise)),
        'n_frames':  int(len(f_noise)),
        'frames':    [row.tolist() for row in f_noise],
    },
}

# ── PreciseOnnxEngine properties ──────────────────────────────────────────────
# Verify the computed properties match JS expectations without loading an ONNX model

class _Props:
    sample_rate = SR
    hop_t       = 0.05
    window_t    = 0.1
    buffer_t    = 1.5

    @property
    def hop_samples(self):    return int(self.sample_rate * self.hop_t + 0.5)
    @property
    def window_samples(self): return int(self.sample_rate * self.window_t + 0.5)
    @property
    def buffer_samples(self):
        raw = int(self.sample_rate * self.buffer_t + 0.5)
        return self.hop_samples * (raw // self.hop_samples)
    @property
    def n_features(self):
        from math import floor
        return 1 + int(floor((self.buffer_samples - self.window_samples) / self.hop_samples))

p = _Props()
vectors['precise_props'] = {
    'window_samples': int(p.window_samples),
    'hop_samples':    int(p.hop_samples),
    'buffer_samples': int(p.buffer_samples),
    'n_features':     int(p.n_features),
    'n_mfcc':         N_MFCC,
}

with open(OUT, 'w') as f:
    json.dump(vectors, f, indent=2)

print(f'Written {OUT}')
print(f'ThresholdDecoder: min_out={td.min_out}, max_out={td.max_out}, cd_len={len(td.cd)}')
print(f'TriggerDetector seq1 results: {res1}')
print(f'MFCC zeros n_frames={len(f_zeros)}, frame[0][:4]={f_zeros[0][:4]}')
print(f'MFCC sine  n_frames={len(f_sine)},  frame[0][:4]={f_sine[0][:4]}')
print(f'MFCC noise n_frames={len(f_noise)}, frame[0][:4]={f_noise[0][:4]}')
print(f'PreciseProps: window={p.window_samples}, hop={p.hop_samples}, buf={p.buffer_samples}, n_feat={p.n_features}')
