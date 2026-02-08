#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SampleData:
    path: Path
    keyit_feat: torch.Tensor  # [1, 1, 105, T]
    target_probs: torch.Tensor  # [1, 24]


def list_audio_files(root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
    files.sort()
    return files


def read_csv_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
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


def run_keyit_features(cli: Path, audio: Path, max_seconds: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        feat_path = Path(tf.name)
    try:
        cmd = [
            str(cli),
            "--input",
            str(audio),
            "--ml-cpu-only",
            "--max-seconds",
            str(max_seconds),
            "--dump-features",
            str(feat_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"keyit-cli failed for {audio}:\n{proc.stderr}\n{proc.stdout}")
        return read_csv_matrix(feat_path)
    finally:
        feat_path.unlink(missing_ok=True)


def run_keyit_probs(cli: Path, audio: Path, max_seconds: float, calibration_file: Path | None) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        probs_path = Path(tf.name)
    try:
        cmd = [
            str(cli),
            "--input",
            str(audio),
            "--ml-cpu-only",
            "--max-seconds",
            str(max_seconds),
            "--dump-probs",
            str(probs_path),
        ]
        if calibration_file is not None:
            cmd += ["--cqt-log-bias-calibration", str(calibration_file)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"keyit-cli failed for {audio}:\n{proc.stderr}\n{proc.stdout}")
        return read_prob_lines(probs_path)
    finally:
        probs_path.unlink(missing_ok=True)


def compute_python_features(audio: Path, max_seconds: float) -> np.ndarray:
    y, _ = librosa.load(audio, sr=44100, mono=True)
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


def load_keynet(checkpoint: Path):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((repo / "third_party" / "MusicalKeyCNN").resolve()))
    from model import KeyNet

    model = KeyNet(num_classes=24, in_channels=1, Nf=20, p=0.0).eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit per-bin CQT log-bias calibration by minimizing output divergence to MusicalKeyCNN"
    )
    p.add_argument("--dataset-dir", type=Path, default=Path("../beatit/training"))
    p.add_argument("--keyit-cli", type=Path, default=Path("build/keyit-cli"))
    p.add_argument("--checkpoint", type=Path, default=Path("third_party/MusicalKeyCNN/checkpoints/keynet.pt"))
    p.add_argument("--out", type=Path, default=Path("scripts/cqt_log_bias_calibration_output.csv"))
    p.add_argument("--max-files", type=int, default=0, help="0 means all files")
    p.add_argument("--max-seconds", type=float, default=8.0 * 60.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--l2", type=float, default=1e-3)
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
    samples: list[SampleData] = []
    for audio in files:
        py_feat = compute_python_features(audio, args.max_seconds)
        keyit_feat = run_keyit_features(args.keyit_cli, audio, args.max_seconds)
        if py_feat.size == 0 or keyit_feat.size == 0:
            continue
        n = min(py_feat.shape[1], keyit_feat.shape[1])
        py_feat = py_feat[:, :n]
        keyit_feat = keyit_feat[:, :n]

        with torch.no_grad():
            py_x = torch.from_numpy(py_feat).unsqueeze(0).unsqueeze(0)
            py_probs = F.softmax(model(py_x), dim=1)

        samples.append(
            SampleData(
                path=audio,
                keyit_feat=torch.from_numpy(keyit_feat).unsqueeze(0).unsqueeze(0),
                target_probs=py_probs,
            )
        )

    if not samples:
        print("No usable samples after preprocessing")
        return 2

    bias = torch.zeros((105,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=args.lr)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32)
        for s in samples:
            x = s.keyit_feat + bias.view(1, 1, 105, 1)
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            kl = F.kl_div(log_probs, s.target_probs, reduction="batchmean")
            total_loss = total_loss + kl
        total_loss = total_loss / len(samples)
        reg = args.l2 * torch.mean(bias * bias)
        loss = total_loss + reg
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            print(
                f"epoch {epoch + 1:4d} "
                f"loss={float(loss.detach()):.6f} "
                f"kl={float(total_loss.detach()):.6f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_vec = bias.detach().cpu().numpy().astype(np.float32)
    args.out.write_text(",".join(f"{x:.9f}" for x in out_vec.tolist()) + "\n")
    print(f"Wrote calibration vector: {args.out}")

    # Evaluate with actual keyit-cli outputs
    mae_base: list[float] = []
    mae_cal: list[float] = []
    top1_base = 0
    top1_cal = 0
    for s in samples:
        py_probs = s.target_probs.squeeze(0).cpu().numpy()
        base_probs = run_keyit_probs(args.keyit_cli, s.path, args.max_seconds, calibration_file=None)
        cal_probs = run_keyit_probs(args.keyit_cli, s.path, args.max_seconds, calibration_file=args.out)
        mae_base.append(float(np.mean(np.abs(py_probs - base_probs))))
        mae_cal.append(float(np.mean(np.abs(py_probs - cal_probs))))
        top1_base += int(int(np.argmax(py_probs)) == int(np.argmax(base_probs)))
        top1_cal += int(int(np.argmax(py_probs)) == int(np.argmax(cal_probs)))

    n = len(samples)
    print("\nEvaluation summary:")
    print(f"  samples: {n}")
    print(f"  prob MAE baseline:   {np.mean(mae_base):.6f}")
    print(f"  prob MAE calibrated: {np.mean(mae_cal):.6f}")
    print(f"  top1 baseline:       {top1_base}/{n}")
    print(f"  top1 calibrated:     {top1_cal}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
