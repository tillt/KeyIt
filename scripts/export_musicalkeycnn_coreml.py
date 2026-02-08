#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import torch


class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.elu = torch.nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(self.bn(self.conv(x)))


class KeyNet(torch.nn.Module):
    def __init__(self, num_classes: int = 24, in_channels: int = 1, nf: int = 20):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels, nf, kernel_size=5)
        self.conv2 = BasicConv2d(nf, nf, kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv3 = BasicConv2d(nf, 2 * nf, kernel_size=3)
        self.conv4 = BasicConv2d(2 * nf, 2 * nf, kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.conv5 = BasicConv2d(2 * nf, 4 * nf, kernel_size=3)
        self.conv6 = BasicConv2d(4 * nf, 4 * nf, kernel_size=3)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.conv7 = BasicConv2d(4 * nf, 8 * nf, kernel_size=3)
        self.conv8 = BasicConv2d(8 * nf, 8 * nf, kernel_size=3)
        self.conv9 = BasicConv2d(8 * nf, num_classes, kernel_size=1)
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv6(self.conv5(x)))
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.global_avgpool(x)
        return torch.flatten(x, 1)


class KeyNetCoreMLWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.model(spec)


def export_coreml(checkpoint: Path,
                  out: Path,
                  deployment: str,
                  precision: str,
                  convert_to: str,
                  dynamic_time: bool) -> None:
    model = KeyNet().eval()
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    wrapper = KeyNetCoreMLWrapper(model).eval()
    dummy = torch.randn(1, 1, 105, 100)

    with torch.no_grad():
        _ = wrapper(dummy)

    traced = torch.jit.trace(wrapper, dummy, check_trace=False)

    compute_precision = None
    if convert_to == "mlprogram":
        compute_precision = ct.precision.FLOAT32 if precision == "float32" else ct.precision.FLOAT16

    input_shape = (1, 1, 105, ct.RangeDim(lower_bound=3, upper_bound=20000, default=100)) \
        if dynamic_time else dummy.shape
    mlmodel = ct.convert(
        traced,
        convert_to=convert_to,
        minimum_deployment_target=getattr(ct.target, deployment),
        inputs=[ct.TensorType(name="spec", shape=input_shape)],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=compute_precision,
    )

    mlmodel.author = "MusicalKeyCNN (Korzeniowski & Widmer), converted for keyit"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Musical key classifier returning 24 Camelot classes"
    mlmodel.input_description["spec"] = "Log-CQT tensor with shape [1,1,105,100]"
    mlmodel.output_description["logits"] = "Class logits for 24 Camelot-mapped keys"

    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MusicalKeyCNN checkpoint to CoreML")
    parser.add_argument(
        "--checkpoint",
        default="third_party/MusicalKeyCNN/checkpoints/keynet.pt",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default="models/keynet.mlpackage",
        type=Path,
    )
    parser.add_argument(
        "--deployment",
        default="macOS13",
        choices=["macOS11", "macOS12", "macOS13", "macOS14", "macOS15"],
    )
    parser.add_argument(
        "--precision",
        default="float32",
        choices=["float32", "float16"],
    )
    parser.add_argument(
        "--convert-to",
        default="mlprogram",
        choices=["mlprogram", "neuralnetwork"],
    )
    parser.add_argument(
        "--dynamic-time",
        action="store_true",
        help="Export model with dynamic time dimension (default: fixed 100-frame input)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    export_coreml(
        checkpoint=args.checkpoint,
        out=args.out,
        deployment=args.deployment,
        precision=args.precision,
        convert_to=args.convert_to,
        dynamic_time=args.dynamic_time,
    )


if __name__ == "__main__":
    main()
