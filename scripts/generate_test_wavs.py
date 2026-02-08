#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import wave
from pathlib import Path
from typing import Iterable, List, Sequence


def _sample_count(sample_rate: int, seconds: float) -> int:
    return int(round(sample_rate * seconds))


def _sine(freq_hz: float, sample_rate: int, seconds: float, amp: float = 0.2) -> List[float]:
    n = _sample_count(sample_rate, seconds)
    w = 2.0 * math.pi * freq_hz / sample_rate
    return [amp * math.sin(w * i) for i in range(n)]


def _chord(freqs_hz: Sequence[float], sample_rate: int, seconds: float, amp: float = 0.2) -> List[float]:
    n = _sample_count(sample_rate, seconds)
    if not freqs_hz:
        return [0.0] * n
    ws = [2.0 * math.pi * f / sample_rate for f in freqs_hz]
    inv = amp / float(len(freqs_hz))
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        for w in ws:
            s += math.sin(w * i)
        out[i] = s * inv
    return out


def _linear_fade(signal: List[float], sample_rate: int, fade_seconds: float = 0.02) -> List[float]:
    out = signal[:]
    fade_n = _sample_count(sample_rate, fade_seconds)
    if fade_n <= 0 or fade_n * 2 > len(out):
        return out
    for i in range(fade_n):
        g = i / float(fade_n - 1) if fade_n > 1 else 1.0
        out[i] *= g
        out[-1 - i] *= g
    return out


def _normalize(signal: List[float], peak: float = 0.95) -> List[float]:
    max_abs = max((abs(x) for x in signal), default=0.0)
    if max_abs <= 1e-12:
        return signal[:]
    scale = peak / max_abs
    return [x * scale for x in signal]


def _concat(parts: Sequence[List[float]]) -> List[float]:
    out: List[float] = []
    for p in parts:
        out.extend(p)
    return out


def _write_wav_mono_16(path: Path, signal: List[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for x in signal:
        y = max(-1.0, min(1.0, x))
        pcm = int(round(y * 32767.0))
        if pcm < 0:
            pcm += 1 << 16
        frames.append(pcm & 0xFF)
        frames.append((pcm >> 8) & 0xFF)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def generate_fixtures(out_dir: Path, sample_rate: int) -> Iterable[Path]:
    created: List[Path] = []

    sine_a4 = _linear_fade(_sine(440.0, sample_rate, 10.0, amp=0.2), sample_rate)
    p = out_dir / "sine_a4_10s.wav"
    _write_wav_mono_16(p, sine_a4, sample_rate)
    created.append(p)

    c_major = _linear_fade(_chord([261.625565, 329.627557, 391.995436], sample_rate, 12.0, amp=0.25), sample_rate)
    p = out_dir / "c_major_triad_12s.wav"
    _write_wav_mono_16(p, _normalize(c_major), sample_rate)
    created.append(p)

    a_minor = _linear_fade(_chord([220.0, 261.625565, 329.627557], sample_rate, 12.0, amp=0.25), sample_rate)
    p = out_dir / "a_minor_triad_12s.wav"
    _write_wav_mono_16(p, _normalize(a_minor), sample_rate)
    created.append(p)

    c_part = _chord([261.625565, 329.627557, 391.995436], sample_rate, 10.0, amp=0.24)
    a_part = _chord([220.0, 261.625565, 329.627557], sample_rate, 10.0, amp=0.24)
    key_change = _linear_fade(_concat([c_part, a_part]), sample_rate)
    p = out_dir / "c_major_to_a_minor_20s.wav"
    _write_wav_mono_16(p, _normalize(key_change), sample_rate)
    created.append(p)

    rng = random.Random(42)
    n = _sample_count(sample_rate, 8.0)
    noise = [0.015 * rng.gauss(0.0, 1.0) for _ in range(n)]
    noise = _linear_fade(noise, sample_rate)
    p = out_dir / "low_noise_8s.wav"
    _write_wav_mono_16(p, _normalize(noise, peak=0.25), sample_rate)
    created.append(p)

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic WAV fixtures for keyit tests")
    parser.add_argument("--out-dir", type=Path, default=Path("tests/audio"), help="Output directory for generated WAV files")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz")
    args = parser.parse_args()

    created = list(generate_fixtures(args.out_dir, args.sample_rate))
    for path in created:
        print(path)
    print(f"Generated {len(created)} files at {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
