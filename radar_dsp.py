"""Synthetic-signal radar DSP helpers.

These functions are for software-only validation with synthetic raw signals.
They do not prove performance on a real 60GHz radar module or real infants.
"""
import numpy as np


def _as_float_array(signal):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


def estimate_bpm_fft(signal, fs, min_bpm=15, max_bpm=70):
    """Estimate breathing BPM from a synthetic raw signal using FFT peak search."""
    x = _as_float_array(signal)
    if len(x) < 2:
        return 0.0
    x = x - np.mean(x)
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    min_hz, max_hz = min_bpm / 60.0, max_bpm / 60.0
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return 0.0
    peak_idx = np.argmax(spec[mask])
    peak_hz = freqs[mask][peak_idx]
    return float(peak_hz * 60.0)


def detect_apnea(signal, fs, window_s=4.0, amp_threshold_ratio=0.2):
    """Detect synthetic apnea by comparing recent RMS to earlier baseline RMS."""
    x = _as_float_array(signal)
    n = int(window_s * fs)
    if len(x) < max(2, n):
        return False
    recent = x[-n:] - np.mean(x[-n:])
    baseline_region = x[:-n] if len(x) > n else x
    baseline_region = baseline_region - np.mean(baseline_region)
    recent_rms = float(np.sqrt(np.mean(recent ** 2)) + 1e-12)
    baseline_rms = float(np.sqrt(np.mean(baseline_region ** 2)) + 1e-12)
    return bool(recent_rms < amp_threshold_ratio * baseline_rms)


def _high_freq_energy(x, fs, highpass_hz):
    x = _as_float_array(x)
    if len(x) < 2:
        return 0.0
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return float(np.sum(spec[freqs >= highpass_hz]) / max(1, len(x)))


def detect_motion(signal, fs, window_s=2.0, highpass_hz=1.5, threshold_ratio=2.0):
    """Detect synthetic motion from recent high-frequency energy."""
    x = _as_float_array(signal)
    n = int(window_s * fs)
    if len(x) < max(2, n):
        return False
    recent = x[-n:]
    baseline = x[:-n] if len(x) > n else x
    recent_energy = _high_freq_energy(recent, fs, highpass_hz)
    baseline_energy = _high_freq_energy(baseline, fs, highpass_hz) + 1e-12
    return bool(recent_energy > threshold_ratio * baseline_energy)


def process_radar_buffer(signal, fs):
    """Process a synthetic radar buffer into fields used by the app/fusion."""
    bpm = estimate_bpm_fft(signal, fs)
    apnea = detect_apnea(signal, fs)
    motion = detect_motion(signal, fs)
    return {
        "breathing_bpm": round(bpm, 2),
        "breathing_rate": round(0.0 if apnea else bpm, 2),
        "apnea": bool(apnea),
        "motion": bool(motion),
        "movement": 1.0 if motion else 0.1,
    }


# ---- 현실적 전처리: range-bin 선택 + 디트렌드 + 대역통과 ----
def _band_energy(sig, fs, low=0.1, high=1.0):
    x = _as_float_array(sig)
    if len(x) < 4:
        return 0.0
    X = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return float(X[(f >= low) & (f <= high)].sum())


def select_range_bin(bins, fs):
    """여러 거리 bin 중 호흡대역 에너지가 가장 큰 bin 선택."""
    bins = np.asarray(bins, dtype=float)
    if bins.ndim == 1:
        return 0
    return int(np.argmax([_band_energy(bins[i], fs) for i in range(bins.shape[0])]))


def detrend(sig):
    """선형 추세(DC·드리프트) 제거."""
    x = _as_float_array(sig)
    n = len(x)
    if n < 2:
        return x
    t = np.arange(n)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def bandpass_fft(sig, fs, low=0.1, high=1.0):
    """FFT 도메인 대역통과(호흡대역만 통과)."""
    x = _as_float_array(sig)
    n = len(x)
    if n < 4:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    X[(f < low) | (f > high)] = 0
    return np.fft.irfft(X, n=n)


def estimate_bpm_realistic(bins, fs):
    """현실적 파이프라인: range-bin 선택 → 디트렌드 → 대역통과 → FFT 피크. (bpm, bin) 반환."""
    bins = np.asarray(bins, dtype=float)
    if bins.ndim == 1:
        bins = bins[None, :]
    idx = select_range_bin(bins, fs)
    sig = bandpass_fft(detrend(bins[idx]), fs)
    return estimate_bpm_fft(sig, fs), idx
