#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch


def list_audio_files(root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
    files.sort()
    return files


def compute_python_features(path: Path, max_seconds: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=44100, mono=True)
    max_samples = int(round(max_seconds * 44100.0))
    if max_samples > 0 and y.shape[0] > max_samples:
        y = y[:max_samples]
    cqt = librosa.cqt(
        y,
        sr=44100,
        hop_length=8820,
        n_bins=105,
        bins_per_octave=24,
        fmin=65,
    )
    return np.log1p(np.abs(cqt))[:, 0:-2].astype(np.float32)


def read_csv_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if row:
                rows.append([float(x) for x in row])
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def read_prob_lines(path: Path) -> np.ndarray:
    vals: list[float] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(float(line))
    return np.asarray(vals, dtype=np.float32)


def run_keyit(
    cli: Path,
    audio: Path,
    max_seconds: float,
    dump_features: Path,
    dump_probs: Path,
    calibration_file: Path | None,
) -> None:
    cmd = [
        str(cli),
        "--input",
        str(audio),
        "--ml-cpu-only",
        "--max-seconds",
        str(max_seconds),
        "--dump-features",
        str(dump_features),
        "--dump-probs",
        str(dump_probs),
    ]
    if calibration_file is not None:
        cmd += ["--cqt-log-bias-calibration", str(calibration_file)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"keyit-cli failed for {audio}:\n{proc.stderr}\n{proc.stdout}")


def load_keynet(checkpoint: Path):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((repo / "third_party" / "MusicalKeyCNN").resolve()))
    from model import KeyNet

    model = KeyNet(num_classes=24, in_channels=1, Nf=20, p=0.0).eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model


def py_probs_from_features(model, feat: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit and evaluate per-bin CQT log-bias calibration for keyit against librosa/MusicalKeyCNN"
    )
    p.add_argument("--dataset-dir", type=Path, default=Path("../beatit/training"))
    p.add_argument("--keyit-cli", type=Path, default=Path("build/keyit-cli"))
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("third_party/MusicalKeyCNN/checkpoints/keynet.pt"),
    )
    p.add_argument("--out", type=Path, default=Path("scripts/cqt_log_bias_calibration.csv"))
    p.add_argument("--max-files", type=int, default=0, help="0 means all files")
    p.add_argument("--max-seconds", type=float, default=8.0 * 60.0)
    p.add_argument(
        "--extensions",
        type=str,
        default=".wav,.aif,.aiff,.mp3,.m4a,.flac",
        help="Comma-separated list",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exts = {x.strip().lower() for x in args.extensions.split(",") if x.strip()}
    files = list_audio_files(args.dataset_dir, exts)
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print(f"No matching audio files found in {args.dataset_dir}")
        return 1

    model = load_keynet(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    bin_sum = np.zeros((105,), dtype=np.float64)
    bin_count = np.zeros((105,), dtype=np.float64)
    feat_mae_base: list[float] = []
    feat_mae_cal: list[float] = []
    prob_mae_base: list[float] = []
    prob_mae_cal: list[float] = []
    top1_match_base = 0
    top1_match_cal = 0
    processed = 0

    tmp = Path(tempfile.mkdtemp(prefix="keyit_cal_"))
    try:
        # Pass 1: collect feature deltas and fit vector.
        for audio in files:
            py_feat = compute_python_features(audio, args.max_seconds)
            base_feat_csv = tmp / "base_feat.csv"
            base_prob_txt = tmp / "base_prob.txt"
            run_keyit(args.keyit_cli, audio, args.max_seconds, base_feat_csv, base_prob_txt, calibration_file=None)
            cpp_feat = read_csv_matrix(base_feat_csv)
            n = min(py_feat.shape[1], cpp_feat.shape[1])
            if n <= 0:
                continue
            py_feat = py_feat[:, :n]
            cpp_feat = cpp_feat[:, :n]
            d = py_feat - cpp_feat
            bin_sum += np.sum(d, axis=1, dtype=np.float64)
            bin_count += float(n)
            processed += 1

        if processed == 0:
            print("No usable files for calibration.")
            return 2

        bias = np.divide(bin_sum, np.maximum(bin_count, 1.0))
        args.out.write_text(",".join(f"{x:.9f}" for x in bias) + "\n")
        print(f"Wrote calibration vector: {args.out}")

        # Pass 2: evaluate baseline vs calibrated.
        for audio in files:
            py_feat = compute_python_features(audio, args.max_seconds)
            py_prob = py_probs_from_features(model, py_feat)

            base_feat_csv = tmp / "base_feat.csv"
            base_prob_txt = tmp / "base_prob.txt"
            run_keyit(args.keyit_cli, audio, args.max_seconds, base_feat_csv, base_prob_txt, calibration_file=None)
            base_feat = read_csv_matrix(base_feat_csv)
            base_prob = read_prob_lines(base_prob_txt)

            cal_feat_csv = tmp / "cal_feat.csv"
            cal_prob_txt = tmp / "cal_prob.txt"
            run_keyit(
                args.keyit_cli,
                audio,
                args.max_seconds,
                cal_feat_csv,
                cal_prob_txt,
                calibration_file=args.out,
            )
            cal_feat = read_csv_matrix(cal_feat_csv)
            cal_prob = read_prob_lines(cal_prob_txt)

            n_base = min(py_feat.shape[1], base_feat.shape[1])
            n_cal = min(py_feat.shape[1], cal_feat.shape[1])
            if n_base > 0:
                feat_mae_base.append(float(np.mean(np.abs(py_feat[:, :n_base] - base_feat[:, :n_base]))))
            if n_cal > 0:
                feat_mae_cal.append(float(np.mean(np.abs(py_feat[:, :n_cal] - cal_feat[:, :n_cal]))))

            if py_prob.shape == base_prob.shape:
                prob_mae_base.append(float(np.mean(np.abs(py_prob - base_prob))))
                top1_match_base += int(np.argmax(py_prob) == np.argmax(base_prob))
            if py_prob.shape == cal_prob.shape:
                prob_mae_cal.append(float(np.mean(np.abs(py_prob - cal_prob))))
                top1_match_cal += int(np.argmax(py_prob) == np.argmax(cal_prob))

        total = len(files)
        print("\nEvaluation summary:")
        print(f"  files considered: {total}")
        print(f"  feature MAE baseline:   {np.mean(feat_mae_base):.6f}")
        print(f"  feature MAE calibrated: {np.mean(feat_mae_cal):.6f}")
        print(f"  prob MAE baseline:      {np.mean(prob_mae_base):.6f}")
        print(f"  prob MAE calibrated:    {np.mean(prob_mae_cal):.6f}")
        print(f"  top1 match baseline:    {top1_match_base}/{total}")
        print(f"  top1 match calibrated:  {top1_match_cal}/{total}")
    finally:
        for p in tmp.glob("*"):
            p.unlink(missing_ok=True)
        tmp.rmdir()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
