# keyit

`keyit` is a macOS musical key detector using CoreML + Accelerate, based on a CoreML port of `MusicalKeyCNN`.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

Build outputs:
- `libkeyit` (`keyit` target)
- `keyit-cli`
- `KeyIt.framework` (on macOS builds in this repo)

## Model export / conversion

Use Python (`torch`, `coremltools`) to export the model:

```bash
python3 scripts/export_musicalkeycnn_coreml.py \
  --checkpoint third_party/MusicalKeyCNN/checkpoints/keynet.pt \
  --out models/keynet.mlpackage

xcrun coremlc compile models/keynet.mlpackage models
```

This produces `models/keynet.mlmodelc`.

Default export is fixed-time for stable execution:
- input: `spec` shape `[1, 1, 105, 100]`
- output: `logits` shape `[1, 24]` or `[24]` (runtime handles both)

`--dynamic-time` export is available but fixed-time is the default/recommended path.

## Model resolution at runtime

If `--model` is provided and exists, it is used first. Otherwise runtime falls back through:
1. model bundled in `KeyIt.framework` resources (`keynet.mlmodelc` / `keynet.mlpackage`)
2. local defaults (`models/keynet.mlmodelc`, `models/keynet.mlpackage`)
3. system-style install paths (`/opt/homebrew/share/keyit/keynet.mlmodelc`, `/usr/local/share/keyit/keynet.mlmodelc`)
4. app bundle resources (`keynet.mlmodelc` / `keynet.mlpackage`)

## CLI

Example:

```bash
./build/keyit-cli --input /path/to/song.mp3 --model models/keynet.mlmodelc
```

Options:
- `-i, --input <path>` audio input (required)
- `-m, --model <path>` model path (`.mlmodelc` or `.mlpackage`)
- `--ml-input <name>` CoreML input feature name (default `spec`)
- `--ml-output <name>` CoreML output feature name (default `logits`)
- `--max-seconds <sec>` cap analyzed duration from start (default `480`)
- `--ml-cpu-only` force CPU-only CoreML execution
- `--cqt-log-bias-calibration <path>` per-bin float list for CQT log-bias calibration
- `--dump-features <path>` write CQT features as CSV (rows = bins)
- `--dump-probs <path>` write probabilities as 24 lines
- `--bench` print benchmark timings
- `--verbose` verbose diagnostics
- `-h, --help` usage

CLI output includes:
- top prediction (`Class`, `Camelot`, `Key`, `Confidence`)
- ambiguity status (`Ambiguous`, `margin`)
- alternate class/camelot/key when ambiguous
- top-5 class probabilities

## Analysis pipeline (current)

Core path:
1. decode to mono float32 (`AVFoundation`)
2. resample to target rate (default `44.1kHz`)
3. compute log-CQT-like matrix (`105` bins by default)
4. trim trailing frames (default `2`)
5. harmonic time-median enhancement (enabled by default)
6. windowed CoreML inference (default window/hop `100/100`)
7. probability aggregation with frame-energy gating
8. optional section-voted probability blending (enabled by default)
9. ambiguity margin / alternate prediction derivation

Advanced knobs are exposed in `keyit::KeyitConfig` (`include/keyit/keyit.h`).

## Library API

Public API:
- `keyit::estimate_key_from_samples(...)`
- `keyit::compute_log_cqt_features_from_samples(...)`
- `keyit::camelot_label(...)`
- `keyit::key_name_label(...)`

See `include/keyit/keyit.h` for config and result structures (`KeyitConfig`, `KeyEstimate`, `PipelineTiming`).

## Parity check

Compare `keyit` against Python reference:

```bash
NUMBA_CACHE_DIR=/tmp ../beatit/.venv/bin/python scripts/parity_check.py \
  --audio /path/to/song.wav \
  --checkpoint third_party/MusicalKeyCNN/checkpoints/keynet.pt \
  --keyit-cli build/keyit-cli \
  --model models/keynet.mlmodelc
```

Reports feature/probability deltas and top-1 agreement.

## Test fixtures

Generate deterministic WAV fixtures:

```bash
python3 scripts/generate_test_wavs.py --out-dir tests/audio
```

Generated fixtures:
- `sine_a4_10s.wav`
- `c_major_triad_12s.wav`
- `b_major_triad_12s.wav`
- `e_major_triad_12s.wav`
- `a_minor_triad_12s.wav`
- `c_major_to_a_minor_20s.wav`
- `low_noise_8s.wav`

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Default test set includes CPU pipeline tests and GPU-path checks (`keyit_gpu_tests`, with skip-on-unsupported behavior configured in CTest).
`keyit_pipeline_tests` also includes synthetic B-side major checks (top-1 expected in Camelot B classes).

## Credits

`keyit` builds on ideas and model work from `MusicalKeyCNN`.

Please cite and refer to the original publication for scientific use and further reading:

[1] F. Korzeniowski and G. Widmer. "Genre-Agnostic Key Classification With Convolutional Neural Networks". In: Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR) (2018) arXiv

[2] F. Korzeniowski and G. Widmer. "End-to-End Musical Key Estimation Using a Convolutional Neural Network". In: Proceedings of the 25th European Signal Processing Conference (2017) arXiv
