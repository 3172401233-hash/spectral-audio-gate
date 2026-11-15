"""Spectral noise gate for mono float audio.

STFT-based noise gate: per-band noise-floor estimation, soft-knee gain
mapping with attack/release smoothing, phase-preserving reconstruction
via overlap-add. Supports both one-shot and streaming (block) processing.
"""

from __future__ import annotations

import numpy as np

_WINDOWS = {
    "hann": lambda n: 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n),
    "hamming": lambda n: 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / n),
    "blackman": lambda n: (
        0.42
        - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
        + 0.08 * np.cos(4.0 * np.pi * np.arange(n) / n)
    ),
    "rect": lambda n: np.ones(n),
}


def _time_coeff(hop_size: int, sample_rate: float, time_ms: float) -> float:
    """Per-frame smoothing coefficient for a given time constant."""
    tau = max(time_ms, 1e-6) / 1000.0
    return 1.0 - np.exp(-hop_size / (sample_rate * tau))


class SpectralGate:
    """Streaming spectral noise gate.

    Parameters
    ----------
    sample_rate : float
        Sample rate in Hz.
    fft_size : int
        Analysis window length in samples.
    hop_size : int
        Hop between consecutive frames (defaults to fft_size // 4).
    window : str or array-like
        Analysis/synthesis window: "hann" (default), "hamming",
        "blackman", "rect", or a custom array of length fft_size.
    num_bands : int
        Number of log-spaced frequency bands between fmin and Nyquist.
    fmin : float
        Lowest band edge in Hz.
    threshold_db : float
        Level above the estimated noise floor at which the gate opens.
    knee_db : float
        Soft-knee width: gain ramps from min_gain to unity over this
        many dB above the threshold.
    min_gain_db : float
        Maximum attenuation applied to noise-only bins. Kept finite so
        instrument tails are ducked rather than hard-zeroed.
    attack_ms, release_ms : float
        Gain envelope time constants (opening / closing).
    noise_rise_ms, noise_fall_ms : float
        Noise-floor estimator time constants. The floor rises slowly
        (so sustained sounds are eventually treated as noise) and falls
        fast (min-statistics style tracking).
    """

    def __init__(
        self,
        sample_rate: float,
        fft_size: int = 1024,
        hop_size: int | None = None,
        window: str | np.ndarray = "hann",
        num_bands: int = 6,
        fmin: float = 50.0,
        threshold_db: float = 12.0,
        knee_db: float = 6.0,
        min_gain_db: float = -24.0,
        attack_ms: float = 10.0,
        release_ms: float = 150.0,
        noise_rise_ms: float = 2000.0,
        noise_fall_ms: float = 50.0,
    ):
        if hop_size is None:
            hop_size = fft_size // 4

        if not isinstance(fft_size, (int, np.integer)) or fft_size < 8:
            raise ValueError("fft_size must be an integer >= 8")
        if not isinstance(hop_size, (int, np.integer)) or not 1 <= hop_size <= fft_size:
            raise ValueError("hop_size must be an integer in [1, fft_size]")
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(num_bands, (int, np.integer)) or num_bands < 1:
            raise ValueError("num_bands must be an integer >= 1")
        nyquist = sample_rate / 2.0
        if not 0 < fmin < nyquist:
            raise ValueError("fmin must be in (0, sample_rate / 2)")
        if threshold_db < 0:
            raise ValueError("threshold_db must be >= 0")
        if knee_db <= 0:
            raise ValueError("knee_db must be > 0")
        if min_gain_db > 0:
            raise ValueError("min_gain_db must be <= 0")
        for name, value in (
            ("attack_ms", attack_ms),
            ("release_ms", release_ms),
            ("noise_rise_ms", noise_rise_ms),
            ("noise_fall_ms", noise_fall_ms),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")

        if isinstance(window, str):
            if window not in _WINDOWS:
                raise ValueError(
                    f"unknown window {window!r}; choose from {sorted(_WINDOWS)} "
                    "or pass a custom array"
                )
            win = _WINDOWS[window](fft_size)
        else:
            win = np.asarray(window, dtype=np.float64)
            if win.shape != (fft_size,):
                raise ValueError("custom window must have length fft_size")
        if not np.all(np.isfinite(win)):
            raise ValueError("window contains non-finite values")

        # Verify the window/hop combination satisfies the COLA constraint:
        # every output position must receive non-zero overlap weight.
        ws = win * win
        norm_pattern = np.zeros(hop_size)
        for p in range(hop_size):
            norm_pattern[p] = ws[p::hop_size].sum()
        if norm_pattern.min() <= 1e-8:
            raise ValueError(
                "window/hop combination is not invertible (COLA violated); "
                "use a smaller hop or a different window"
            )

        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self._window = win
        self._ws = ws

        # Log-spaced band edges; every rfft bin is assigned to a band.
        edges = np.geomspace(fmin, nyquist, num_bands + 1)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        self._bin_band = np.clip(
            np.searchsorted(edges, freqs, side="right") - 1, 0, num_bands - 1
        )
        self.num_bands = int(num_bands)

        self._thresh_lin = 10.0 ** (threshold_db / 20.0)
        self._knee_db = float(knee_db)
        self._min_gain_db = float(min_gain_db)

        self._attack = _time_coeff(hop_size, sample_rate, attack_ms)
        self._release = _time_coeff(hop_size, sample_rate, release_ms)
        self._noise_rise = _time_coeff(hop_size, sample_rate, noise_rise_ms)
        self._noise_fall = _time_coeff(hop_size, sample_rate, noise_fall_ms)

        # Envelope / noise-floor state (persists across blocks).
        self._floor: np.ndarray | None = None
        self._gain = np.ones(self.num_bands)

        # Streaming buffers. Absolute sample positions:
        # _in_start   : position of _in_buf[0]
        # _frame_start: start of the next frame to analyze
        # _out_start  : position of _out_buf[0] / _norm_buf[0]
        # _emit_pos   : next position to emit
        self._in_buf = np.zeros(0)
        self._in_start = 0
        self._frame_start = 0
        self._out_buf = np.zeros(0)
        self._norm_buf = np.zeros(0)
        self._out_start = 0
        self._emit_pos = 0

    @property
    def latency(self) -> int:
        """Worst-case streaming latency in samples (fft_size - 1)."""
        return self.fft_size - 1

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def process_block(self, samples) -> np.ndarray:
        """Feed a block of mono samples; return newly available output."""
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("input must be a 1-D mono signal")
        if not np.all(np.isfinite(x)):
            raise ValueError("input contains non-finite values")

        self._in_buf = np.concatenate([self._in_buf, x])
        while self._in_start + self._in_buf.size - self._frame_start >= self.fft_size:
            offset = self._frame_start - self._in_start
            self._process_frame(self._in_buf[offset : offset + self.fft_size])
            self._frame_start += self.hop_size
        return self._emit(self._frame_start)

    def flush(self) -> np.ndarray:
        """Process remaining buffered input and emit the window tail."""
        total = self._in_start + self._in_buf.size
        if total == 0 and self._frame_start == 0:
            return np.zeros(0)
        while self._frame_start < total:
            need = self._frame_start + self.fft_size - self._in_start - self._in_buf.size
            if need > 0:
                self._in_buf = np.concatenate([self._in_buf, np.zeros(need)])
            offset = self._frame_start - self._in_start
            self._process_frame(self._in_buf[offset : offset + self.fft_size])
            self._frame_start += self.hop_size
        end = self._frame_start - self.hop_size + self.fft_size
        return self._emit(end)

    def process(self, samples) -> np.ndarray:
        """One-shot processing of a complete signal."""
        parts = [self.process_block(samples), self.flush()]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.zeros(0)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> None:
        spectrum = np.fft.rfft(frame * self._window)
        mag = np.abs(spectrum)

        # Per-band RMS magnitude: robust to tonal peaks being diluted by
        # wide bands (unlike a plain mean).
        counts = np.bincount(self._bin_band, minlength=self.num_bands)
        sums = np.bincount(self._bin_band, weights=mag * mag, minlength=self.num_bands)
        band_mag = np.sqrt(sums / np.maximum(counts, 1))

        # Noise floor: asymmetric exponential tracking per band.
        if self._floor is None:
            self._floor = band_mag.copy()
        else:
            coeff = np.where(band_mag < self._floor, self._noise_fall, self._noise_rise)
            self._floor += coeff * (band_mag - self._floor)

        # Soft-knee gain target from level above the (scaled) floor.
        thresh = np.maximum(self._floor * self._thresh_lin, 1e-12)
        over_db = 20.0 * np.log10(np.maximum(band_mag / thresh, 1e-12))
        frac = np.clip(over_db / self._knee_db, 0.0, 1.0)
        target = 10.0 ** (self._min_gain_db * (1.0 - frac) / 20.0)

        # Attack/release envelope smoothing per band.
        coeff = np.where(target > self._gain, self._attack, self._release)
        self._gain += coeff * (target - self._gain)

        # Real non-negative per-bin gain: magnitudes are scaled, phase kept.
        out = np.fft.irfft(spectrum * self._gain[self._bin_band], n=self.fft_size)
        out *= self._window

        start = self._frame_start - self._out_start
        need = start + self.fft_size - self._out_buf.size
        if need > 0:
            self._out_buf = np.concatenate([self._out_buf, np.zeros(need)])
            self._norm_buf = np.concatenate([self._norm_buf, np.zeros(need)])
        self._out_buf[start : start + self.fft_size] += out
        self._norm_buf[start : start + self.fft_size] += self._ws

    def _emit(self, end: int) -> np.ndarray:
        """Emit finalized positions [self._emit_pos, end)."""
        n = end - self._emit_pos
        if n <= 0:
            return np.zeros(0)
        offset = self._emit_pos - self._out_start
        norm = self._norm_buf[offset : offset + n]
        # Boundary positions with (near-)zero window weight carry no signal.
        out = np.where(
            norm > 1e-12,
            self._out_buf[offset : offset + n] / np.maximum(norm, 1e-12),
            0.0,
        )
        self._emit_pos = end

        # Drop consumed prefixes to keep buffers bounded.
        self._out_buf = self._out_buf[offset + n :]
        self._norm_buf = self._norm_buf[offset + n :]
        self._out_start = self._emit_pos
        drop_in = self._frame_start - self._in_start
        if drop_in > 0:
            self._in_buf = self._in_buf[drop_in:]
            self._in_start = self._frame_start
        return out
