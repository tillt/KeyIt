# KeyIT

<p align="center"><img height="340px" src="site/images/KeyIT_Logo_1254x1254.png" /></p>

KeyIT is a macOS musical key detector using CoreML + Accelerate, based on a CoreML port of `MusicalKeyCNN`.

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

## Credits

KeyIT builds on ideas and model work from `MusicalKeyCNN`.

Please cite and refer to the original publication for scientific use and further reading:

[1] F. Korzeniowski and G. Widmer. "Genre-Agnostic Key Classification With Convolutional Neural Networks". In: Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR) (2018) arXiv

[2] F. Korzeniowski and G. Widmer. "End-to-End Musical Key Estimation Using a Convolutional Neural Network". In: Proceedings of the 25th European Signal Processing Conference (2017) arXiv
