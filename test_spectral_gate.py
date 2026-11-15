import unittest

import numpy as np

from spectral_gate import SpectralGate

SR = 8000
FFT = 256
HOP = 64


def make_gate(**overrides):
    params = dict(
        sample_rate=SR,
        fft_size=FFT,
        hop_size=HOP,
        num_bands=6,
        attack_ms=5.0,
        release_ms=100.0,
        min_gain_db=-24.0,
    )
    params.update(overrides)
    return SpectralGate(**params)


def rms(x):
    return np.sqrt(np.mean(np.asarray(x) ** 2)) if len(x) else 0.0


class TestBlockConsistency(unittest.TestCase):
    def test_chunked_matches_one_shot(self):
        rng = np.random.default_rng(42)
        t = np.arange(5000) / SR
        x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(len(t))

        ref = make_gate().process(x)
        for block_size in (1, 7, HOP, 100, 1000, 4096):
            gate = make_gate()
            outs = [
                gate.process_block(x[i : i + block_size])
                for i in range(0, len(x), block_size)
            ]
            outs.append(gate.flush())
            got = np.concatenate([o for o in outs if o.size])
            self.assertEqual(got.shape, ref.shape)
            np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)

    def test_output_length(self):
        x = np.zeros(1000)
        y = make_gate().process(x)
        last_start = (len(x) - 1) // HOP * HOP
        self.assertEqual(len(y), last_start + FFT)


class TestGateOpenClose(unittest.TestCase):
    def test_steady_noise_is_attenuated(self):
        rng = np.random.default_rng(1)
        noise = 0.1 * rng.standard_normal(SR)  # 1 s of steady noise
        y = make_gate().process(noise)
        tail_in = noise[SR // 2 :]
        tail_out = y[SR // 2 : SR // 2 + len(tail_in)]
        # min_gain is -24 dB (~0.06); allow generous margin.
        self.assertLess(rms(tail_out), 0.3 * rms(tail_in))

    def test_gate_reopens_for_sudden_voice(self):
        rng = np.random.default_rng(2)
        noise = 0.05 * rng.standard_normal(SR)
        t = np.arange(SR // 4) / SR
        voice = 0.5 * np.sin(2 * np.pi * 1000 * t)
        x = np.concatenate([noise, voice])
        y = make_gate().process(x)

        noise_out = y[int(0.5 * SR) : int(0.9 * SR)]
        self.assertLess(rms(noise_out), 0.3 * rms(noise[int(0.5 * SR) : int(0.9 * SR)]))

        # After the attack time the tone must come through nearly intact.
        seg = slice(SR + int(0.05 * SR), SR + int(0.2 * SR))
        self.assertGreater(rms(y[seg]), 0.7 * rms(voice[int(0.05 * SR) : int(0.2 * SR)]))

    def test_short_impulse_passes(self):
        x = np.zeros(4000)
        x[2000] = 0.8
        y = make_gate().process(x)
        self.assertGreater(np.max(np.abs(y)), 0.5)


class TestBandIndependence(unittest.TestCase):
    def test_noise_in_one_band_does_not_gate_other_bands(self):
        rng = np.random.default_rng(3)
        n = SR + SR // 2
        # Low-band rumble: one-pole lowpass of white noise, below ~250 Hz.
        white = rng.standard_normal(n)
        low = np.zeros(n)
        for i in range(1, n):
            low[i] = 0.98 * low[i - 1] + 0.02 * white[i]
        low *= 0.3 / rms(low)
        t = np.arange(n) / SR
        tone = 0.4 * np.sin(2 * np.pi * 3000 * t)
        tone[: SR // 2] = 0.0  # noise-only learn period, then tone joins
        x = low + tone

        # Freeze the floor's upward adaptation so the sustained tone is not
        # absorbed into the noise estimate during the measurement.
        y = make_gate(noise_rise_ms=1e9).process(x)
        steady = slice(int(0.8 * SR), int(1.4 * SR))
        seg_len = steady.stop - steady.start
        freqs = np.fft.rfftfreq(seg_len, 1 / SR)
        band_lo = freqs < 500
        band_hi = (freqs > 2500) & (freqs < 3500)
        spec_in = np.abs(np.fft.rfft(x[steady]))
        spec_out = np.abs(np.fft.rfft(y[steady]))

        self.assertLess(spec_out[band_lo].sum(), 0.4 * spec_in[band_lo].sum())
        self.assertGreater(spec_out[band_hi].sum(), 0.7 * spec_in[band_hi].sum())


class TestPhasePreservation(unittest.TestCase):
    def test_tone_passes_with_zero_phase_shift(self):
        # Bin-centered tone; keep the noise floor frozen so the gain
        # converges to exactly 1 and stays there.
        k = 20
        freq = k * SR / FFT
        n = 4096
        t = np.arange(n) / SR
        x = 0.4 * np.sin(2 * np.pi * freq * t)
        x[:256] = 0.0  # silent learn period keeps the floor at zero
        gate = make_gate(noise_rise_ms=1e9, attack_ms=1.0)
        y = gate.process(x)

        seg = slice(2048, 3072)  # steady state, attack fully settled
        ref = x[seg]
        out = y[seg]
        scale = np.dot(out, ref) / np.dot(ref, ref)
        residual = out - scale * ref
        self.assertGreater(scale, 0.99)
        self.assertLess(rms(residual), 1e-3 * rms(ref))


class TestSilence(unittest.TestCase):
    def test_silence_stays_silent(self):
        y = make_gate().process(np.zeros(5000))
        self.assertTrue(np.all(y == 0.0))

    def test_empty_input(self):
        y = make_gate().process(np.zeros(0))
        self.assertEqual(len(y), 0)


class TestInvalidParams(unittest.TestCase):
    def test_bad_params_raise(self):
        bad = [
            dict(sample_rate=0),
            dict(sample_rate=-8000),
            dict(fft_size=4),
            dict(fft_size=100.5),
            dict(hop_size=0),
            dict(hop_size=FFT + 1),
            dict(num_bands=0),
            dict(fmin=0),
            dict(fmin=SR),  # above Nyquist
            dict(threshold_db=-1),
            dict(knee_db=0),
            dict(min_gain_db=3),
            dict(attack_ms=0),
            dict(release_ms=-5),
            dict(noise_rise_ms=0),
            dict(noise_fall_ms=-1),
            dict(window="nuttall"),
            dict(window=np.ones(FFT + 1)),
            dict(window="hann", hop_size=FFT),  # COLA violated at boundaries
        ]
        for params in bad:
            with self.subTest(params=params):
                kwargs = dict(sample_rate=SR, fft_size=FFT, hop_size=HOP)
                kwargs.update(params)
                with self.assertRaises(ValueError):
                    SpectralGate(**kwargs)

    def test_bad_input_raises(self):
        gate = make_gate()
        with self.assertRaises(ValueError):
            gate.process_block(np.zeros((10, 2)))
        with self.assertRaises(ValueError):
            gate.process_block(np.array([np.nan, 1.0]))


if __name__ == "__main__":
    unittest.main()
