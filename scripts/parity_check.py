#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import librosa
import numpy as np
import torch

import sys


def load_keynet(checkpoint: Path):
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "third_party" / "MusicalKeyCNN").resolve()))
    from model import KeyNet

    model = KeyNet(num_classes=24, in_channels=1, Nf=20, p=0.0).eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model


def compute_python_features(audio_path: Path):
    y, _ = librosa.load(audio_path, sr=44100, mono=True)
    cqt = librosa.cqt(y, sr=44100, hop_length=8820, n_bins=105, bins_per_octave=24, fmin=65)
    spec = np.log1p(np.abs(cqt))
    spec = spec[:, 0:-2]
    return spec.astype(np.float32)


def read_csv_matrix(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def read_prob_lines(path: Path):
    vals = []
    with path.open("r") as f:
        for line in f:
            t = line.strip()
            if t:
                vals.append(float(t))
    return np.asarray(vals, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare keyit C++ pipeline against MusicalKeyCNN python reference")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--checkpoint", default=Path("third_party/MusicalKeyCNN/checkpoints/keynet.pt"), type=Path)
    parser.add_argument("--keyit-cli", default=Path("build/keyit-cli"), type=Path)
    parser.add_argument("--model", default=Path("models/keynet.mlmodelc"), type=Path)
    parser.add_argument("--work", default=Path("/tmp/keyit_parity"), type=Path)
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    cpp_feat = args.work / "cpp_features.csv"
    cpp_probs = args.work / "cpp_probs.txt"

    py_feat = compute_python_features(args.audio)
    model = load_keynet(args.checkpoint)

    x = torch.from_numpy(py_feat).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        py_logits = model(x)
        py_probs = torch.softmax(py_logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    cmd = [
        str(args.keyit_cli),
        "--input", str(args.audio),
        "--model", str(args.model),
        "--dump-features", str(cpp_feat),
        "--dump-probs", str(cpp_probs),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("keyit-cli failed")
        print(proc.stdout)
        print(proc.stderr)
        return proc.returncode

    cpp_feat_arr = read_csv_matrix(cpp_feat)
    cpp_probs_arr = read_prob_lines(cpp_probs)

    print(f"Python features: {py_feat.shape}")
    print(f"C++ features:    {cpp_feat_arr.shape}")
    if py_feat.shape == cpp_feat_arr.shape and py_feat.size > 0:
        feat_diff = np.abs(py_feat - cpp_feat_arr)
        print(f"Feature MAE:     {feat_diff.mean():.8f}")
        print(f"Feature MaxAbs:  {feat_diff.max():.8f}")
    else:
        print("Feature shape mismatch, cannot compute error")

    print(f"Python probs:    {py_probs.shape}")
    print(f"C++ probs:       {cpp_probs_arr.shape}")
    if py_probs.shape == cpp_probs_arr.shape and py_probs.size > 0:
        prob_diff = np.abs(py_probs - cpp_probs_arr)
        print(f"Prob MAE:        {prob_diff.mean():.8f}")
        print(f"Prob MaxAbs:     {prob_diff.max():.8f}")
        print(f"Top1 python:     {int(np.argmax(py_probs))}")
        print(f"Top1 c++:        {int(np.argmax(cpp_probs_arr))}")
    else:
        print("Probability shape mismatch, cannot compute error")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
