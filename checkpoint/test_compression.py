import csv
import io
from datetime import datetime
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import torch
from evaluator import Evaluator
from functools import partial


def jpeg_compress(input_tensor: torch.Tensor,
                  quality: int = 85) -> bytes:
    """
    Compress a tensor to JPEG bytes.

    Args:
        input_tensor: (1,C,H,W) or (C,H,W) tensor in [0,1]
        quality: JPEG quality (1-100)

    Returns:
        JPEG-encoded bytes
    """

    if input_tensor.ndim == 4:
        input_tensor = input_tensor.squeeze(0)

    img = transforms.ToPILImage()(input_tensor.cpu())

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True
    )

    return buffer.getvalue()

def jpeg_decompress(jpeg_bytes: bytes,
                    device: torch.device = None) -> torch.Tensor:
    """
    Decode JPEG bytes back into a tensor.

    Returns:
        (1,C,H,W) tensor in [0,1]
    """

    buffer = io.BytesIO(jpeg_bytes)

    img = Image.open(buffer).convert("RGB")

    tensor = transforms.ToTensor()(img).unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    return tensor

def jpeg_roundtrip(input_tensor: torch.Tensor,
                   quality: int = 85,
                   device: torch.device = None) -> torch.Tensor:
    """
    Compress and decompress a tensor using JPEG.

    Args:
        input_tensor: (1,C,H,W) or (C,H,W) tensor in [0,1]
        quality: JPEG quality (1-100)
        device: device to return the output tensor on
    Returns:
        (1,C,H,W) tensor in [0,1]
    """
    jpeg_bytes = jpeg_compress(input_tensor, quality=quality)
    output_tensor = jpeg_decompress(jpeg_bytes, device=device)
    return output_tensor

QUALITY_LEVELS = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]


def save_results_to_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    # Test the JPEG roundtrip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = Evaluator(checkpoint_path="outputs/checkpoints_good/ckpt_ep000_s004000.pt")
    results = []

    for quality in QUALITY_LEVELS:
        jpeg_roundtrip_quality = partial(jpeg_roundtrip, quality=quality, device=device)
        print(f"Testing JPEG roundtrip with quality={quality}...")
        summary = evaluator.run(transformation=jpeg_roundtrip_quality)
        print(f"JPEG quality={quality} | summary: {summary}")
        if summary:
            results.append({"quality": quality, **summary})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("outputs") / "metrics" / f"jpeg_compression_results_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(results, output_path)
    print(f"Saved CSV results to {output_path}")


if __name__ == "__main__":
    main()