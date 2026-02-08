# keyit

`keyit` is a macOS musical key detector built around a CoreML port of `MusicalKeyCNN`.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

This builds:
- `libkeyit` (library target: `keyit`)
- `keyit-cli`

## Convert model

Use a Python environment with `torch` and `coremltools`.

```bash
python3 scripts/export_musicalkeycnn_coreml.py \
  --checkpoint third_party/MusicalKeyCNN/checkpoints/keynet.pt \
  --out models/keynet.mlpackage

xcrun coremlc compile models/keynet.mlpackage models
```

This produces `models/keynet.mlmodelc` used by default.

Default export is fixed-shape for stable GPU/MPS execution:
- input `spec`: `[1, 1, 105, 100]`
- output `logits`: `[1, 24]` or `[24]` (flattened by runtime)

Optional dynamic-time export is available with `--dynamic-time`.

## CLI

```bash
./build/keyit-cli --input /path/to/song.mp3 --model models/keynet.mlmodelc
```

Useful options:
- `--ml-input spec`
- `--ml-output logits`
- `--ml-cpu-only` (force CoreML CPU-only execution)
- `--max-seconds 480` (default: 8 minutes)
- `--dump-features /tmp/features.csv`
- `--dump-probs /tmp/probs.txt`
- `--verbose`

## Notes on preprocessing

`keyit` uses a C++ frontend designed to mirror the original `predict_keys.py` contract:
- mono audio
- resample to 44.1 kHz
- CQT-like 105-bin log-frequency frontend
- full-track frame analysis with trailing-frame trim (`[:, 0:-2]`)
- windowed inference over fixed 100-frame chunks with aggregated logits

This keeps inference fully on macOS frameworks (`AVFoundation`, `Accelerate`, `CoreML`) with no runtime Python dependency.

## Parity check

To compare `keyit` against the Python reference implementation:

```bash
NUMBA_CACHE_DIR=/tmp ../beatit/.venv/bin/python scripts/parity_check.py \
  --audio /path/to/song.wav \
  --checkpoint third_party/MusicalKeyCNN/checkpoints/keynet.pt \
  --keyit-cli build/keyit-cli \
  --model models/keynet.mlmodelc
```

This reports feature/probability error metrics and top-1 agreement.

## Generate test WAV fixtures

Create deterministic synthetic WAVs for local tests:

```bash
python3 scripts/generate_test_wavs.py --out-dir tests/audio
```

Generated files include:
- `sine_a4_10s.wav`
- `c_major_triad_12s.wav`
- `a_minor_triad_12s.wav`
- `c_major_to_a_minor_20s.wav`
- `low_noise_8s.wav`

## Tests

Run default tests:

```bash
ctest --test-dir build --output-on-failure
```

`keyit_gpu_tests` runs by default and validates the preferred non-CPU CoreML path.
