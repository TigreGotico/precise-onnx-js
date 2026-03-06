// MFCC feature extraction — exact port of Python's sonopy.mfcc_spec
// Matches ovos-ww-plugin-precise-onnx/inference.py which calls sonopy.mfcc_spec.
//
// Key implementation notes (all verified against sonopy source):
//  - No windowing function: frames are raw rectangular slices (chop_array)
//  - Power spectrum: (re²+im²) / fftSize  (divided by fftSize like sonopy)
//  - Filterbank grid indices: floor(hz * nBins / sampleRate)  where nBins=fftSize/2+1
//  - correct_grid: pushes forward duplicate bin indices to keep filters distinct
//  - safe_log: Math.log(Math.max(x, Number.EPSILON)) — matches np.finfo(float).eps
//  - First MFCC coefficient: replaced by safe_log(sum of all power bins) per frame

'use strict';

// ── FFT (iterative Cooley-Tukey) ──────────────────────────────────────────────

function _fftInPlace(re, im) {
    const n = re.length;
    let j = 0;
    for (let i = 1; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let curRe = 1, curIm = 0;
            for (let k = 0; k < len / 2; k++) {
                const uRe = re[i + k], uIm = im[i + k];
                const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
                const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
                re[i + k]           = uRe + vRe;  im[i + k]           = uIm + vIm;
                re[i + k + len / 2] = uRe - vRe;  im[i + k + len / 2] = uIm - vIm;
                const newRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = newRe;
            }
        }
    }
}

// ── power_spec ────────────────────────────────────────────────────────────────
// Port of sonopy.power_spec(audio, (window, hop), fft_size).
// Returns Float64Array[] — one power spectrum per frame, length = fftSize/2+1.
// Power = (re² + im²) / fftSize  (matches numpy rfft / fftSize convention).
// NO windowing function applied — pure rectangular frames (sonopy does the same).

function powerSpec(audio, windowSize, hopSize, fftSize) {
    const nBins = fftSize / 2 + 1;
    const frames = [];
    // chop_array: [audio[i-win:i] for i in range(win, len+1, hop)]
    for (let i = windowSize; i <= audio.length; i += hopSize) {
        const re = new Float64Array(fftSize);
        const im = new Float64Array(fftSize);
        const slice = audio.subarray ? audio.subarray(i - windowSize, i)
                                     : audio.slice(i - windowSize, i);
        const take = Math.min(fftSize, slice.length);
        for (let k = 0; k < take; k++) re[k] = slice[k];
        _fftInPlace(re, im);
        const power = new Float64Array(nBins);
        for (let k = 0; k < nBins; k++) {
            power[k] = (re[k] * re[k] + im[k] * im[k]) / fftSize;
        }
        frames.push(power);
    }
    return frames;
}

// ── filterbanks ───────────────────────────────────────────────────────────────
// Port of sonopy.filterbanks(sampleRate, numFilt, nBins) with @lru_cache.
// nBins = fftSize/2 + 1.
// Uses 1127*ln(1+hz/700) mel scale and correct_grid for duplicate-free indices.

function _buildFilterbanks(sampleRate, numFilt, nBins) {
    function hzToMel(f)  { return 1127 * Math.log(1 + f / 700); }
    function melToHz(m)  { return 700  * (Math.exp(m / 1127) - 1); }

    const gridMels  = new Float64Array(numFilt + 2);
    const lo = hzToMel(0), hi = hzToMel(sampleRate);
    for (let i = 0; i <= numFilt + 1; i++) {
        gridMels[i] = lo + i * (hi - lo) / (numFilt + 1);
    }

    // Convert to FFT bin indices: int(hz * nBins / sampleRate)
    const rawIdx = new Int32Array(numFilt + 2);
    for (let i = 0; i <= numFilt + 1; i++) {
        rawIdx[i] = Math.trunc(melToHz(gridMels[i]) * nBins / sampleRate);
    }

    // correct_grid: push duplicate indices forward so every filter is distinct
    // Python: for prev, i in zip([x[0]-1]+x, x): offset = max(0, offset+prev+1-i)
    const grid = new Int32Array(numFilt + 2);
    let offset = 0;
    for (let i = 0; i <= numFilt + 1; i++) {
        const prev  = i === 0 ? rawIdx[0] - 1 : rawIdx[i - 1];
        offset = Math.max(0, offset + prev + 1 - rawIdx[i]);
        grid[i] = rawIdx[i] + offset;
    }

    // Build triangular filters using linspace(0,1,n,False) and linspace(1,0,n,False)
    const banks = [];
    for (let m = 0; m < numFilt; m++) {
        const bank   = new Float64Array(nBins);
        const left   = grid[m];
        const middle = grid[m + 1];
        const right  = grid[m + 2];
        // Rising:  linspace(0, 1, middle-left, endpoint=False) → k/n for k in [0,n)
        if (middle > left) {
            const n = middle - left;
            for (let k = 0; k < n && left + k < nBins; k++) bank[left + k] = k / n;
        }
        // Falling: linspace(1, 0, right-middle, endpoint=False) → (n-k)/n for k in [0,n)
        if (right > middle) {
            const n = right - middle;
            for (let k = 0; k < n && middle + k < nBins; k++) bank[middle + k] = (n - k) / n;
        }
        banks.push(bank);
    }
    return banks;
}

// Module-level filterbank cache (equivalent to @lru_cache in sonopy)
const _fbCache = {};
function filterbanks(sampleRate, numFilt, nBins) {
    const key = `${sampleRate}_${numFilt}_${nBins}`;
    if (!_fbCache[key]) _fbCache[key] = _buildFilterbanks(sampleRate, numFilt, nBins);
    return _fbCache[key];
}

// ── safe_log ──────────────────────────────────────────────────────────────────
// Port of sonopy.safe_log: log(clip(x, np.finfo(float).eps, None))
// Number.EPSILON === np.finfo(float).eps === 2.220446049250313e-16

function safeLog(x) { return Math.log(Math.max(x, Number.EPSILON)); }

// ── DCT-II (ortho, port of scipy.fft.dct(x, type=2, norm='ortho')) ──────────
// Matches scipy's normalization factors:
//   k=0: y[0] = sum(x) * sqrt(1/N)
//   k>0: y[k] = 2 * sum(x[n]*cos(π·k·(2n+1)/(2N))) * sqrt(1/(2N))
//            = sum(...) * sqrt(2/N)

function dctII(x, numCoeffs) {
    const N = x.length;
    const out = new Float64Array(numCoeffs);
    for (let k = 0; k < numCoeffs; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++) {
            sum += x[n] * Math.cos(Math.PI * k * (2 * n + 1) / (2 * N));
        }
        out[k] = sum * (k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N));
    }
    return out;
}

// ── mfccSpec ──────────────────────────────────────────────────────────────────
// Port of sonopy.mfcc_spec(audio, sampleRate, (windowSize, hopSize),
//                          fft_size, num_filt, num_coeffs).
//
// Returns Float32Array[] — one row per frame, length numCoeffs.
// Column 0 = safe_log(sum of power spectrum) — same as sonopy's energy replacement.

function mfccSpec(audio, sampleRate, windowSize, hopSize, numFilt, fftSize, numCoeffs) {
    const nBins = fftSize / 2 + 1;   // 257 for fftSize=512

    const powers = powerSpec(audio, windowSize, hopSize, fftSize);
    if (powers.length === 0) return [];

    const banks = filterbanks(sampleRate, numFilt, nBins);

    return powers.map(power => {
        // Mel energies: dot(power, banks.T) then safe_log
        const logMel = new Float64Array(numFilt);
        for (let m = 0; m < numFilt; m++) {
            let dot = 0;
            for (let k = 0; k < nBins; k++) dot += power[k] * banks[m][k];
            logMel[m] = safeLog(dot);
        }

        // DCT-II ortho on log mel energies
        const coeffs = dctII(logMel, numCoeffs);

        // Replace first coefficient with safe_log(total power) — matches sonopy
        let totalPower = 0;
        for (let k = 0; k < nBins; k++) totalPower += power[k];
        coeffs[0] = safeLog(totalPower);

        return new Float32Array(coeffs);
    });
}

if (typeof module !== 'undefined') {
    module.exports = { mfccSpec, powerSpec, filterbanks, dctII, safeLog };
}
