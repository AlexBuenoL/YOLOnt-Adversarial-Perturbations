from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from config import Config, cfg
from dataset import build_stream
from model import build_perturbation_net


# -- perceptual quality helpers --

def psnr(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Peak signal-to-Noise ratio in dB (higher = less distortion)."""
    mse = F.mse_loss(adversarial, original).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)

def ssim(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Simplified single-scale SSIM (averaged over channels).
    Returns a value in [-1, 1]; values close to 1 = almost identical.
    """
    mu1 = F.avg_pool2d(original, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(adversarial, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(original ** 2, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(adversarial ** 2, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(original * adversarial, window_size, 1, window_size // 2) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return (numerator / (denominator + 1e-8)).mean().item()


# -- YOLO helpers --

def _max_person_confidence(result) -> float:
    """
    Extract the highest person-class confidence from a YOLO result object.
    Returns 0.0 if no person detected.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0.0

    person_confs = [
        float(b.conf[0].item())
        for b in boxes
        if int(b.cls[0].item()) == 0
    ]
    return max(person_confs) if person_confs else 0.0

def _has_person_detection(result) -> bool:
    return _max_person_confidence(result) > 0.0


# -- evaluator --

class Evaluator:
    """
    Load a trained perturbation network and evaluate it against YOLO.
    """
    def __init__(
        self,
        config: Config = cfg,
        checkpoint_path: str | Path | None = None,
    ):
        self.cfg    = config
        self.device = torch.device(config.device)

        # perturbation network
        self.net = build_perturbation_net(config).to(self.device)

        if checkpoint_path is not None:
            ckpt = torch.load(Path(checkpoint_path), map_location=self.device)
            self.net.load_state_dict(ckpt["model_state_dict"])
            print(f"[evaluator] Loaded checkpoint: {checkpoint_path}")
        else:
            print("[evaluator] No checkpoint provided --> using random weights.")

        self.net.eval()

        # YOLO (wrapper for easy result parsing)
        self.yolo = YOLO(config.yolo_weights)

        # data stream --> use eval split
        self.stream = build_stream(config, split_type="eval")

    @torch.no_grad()
    def _evaluate_single(
        self, orig_tensor: torch.Tensor, transformation: Callable = None
    ) -> dict:
        """
        Evaluate one image.  
        Returns a dict of metrics per image.
        """
        adv_tensor, _ = self.net(orig_tensor)

        # apply optional post-perturbation transformation to adversarial tensor
        if transformation:
            adv_tensor = transformation(adv_tensor)

        # convert tensors to numpy uint8 for YOLO inference
        def to_numpy(t):
            return (
                t.squeeze(0)
                 .cpu()
                 .permute(1, 2, 0)
                 .mul(255)
                 .byte()
                 .numpy()
            )

        orig_np = to_numpy(orig_tensor)
        adv_np = to_numpy(adv_tensor)

        orig_results = self.yolo(orig_np, verbose=False)
        adv_results = self.yolo(adv_np, verbose=False)

        orig_conf = _max_person_confidence(orig_results[0])
        adv_conf = _max_person_confidence(adv_results[0])

        return {
            "orig_has_person": _has_person_detection(orig_results[0]),
            "adv_has_person": _has_person_detection(adv_results[0]),
            "orig_conf": orig_conf,
            "adv_conf": adv_conf,
            "conf_drop": orig_conf - adv_conf,
            "psnr": psnr(orig_tensor, adv_tensor),
            "ssim": ssim(orig_tensor, adv_tensor),
        }

    def run(self, steps: int | None = None, transformation: Callable = None) -> dict:
        """
        Run evaluation over `steps` images.
        """
        n = steps or self.cfg.eval_steps

        print(f"\n[evaluator] Processing {n} images from eval partition...\n")

        # only count images where YOLO originally detects a person
        valid = 0
        suppressed = 0
        total_conf_drop = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        for i in range(n):
            img = next(self.stream).to(self.device)
            m   = self._evaluate_single(img, transformation=transformation)

            if not m["orig_has_person"]:
                continue

            valid += 1

            if not m["adv_has_person"]:
                suppressed += 1

            total_conf_drop += m["conf_drop"]
            total_psnr += m["psnr"]
            total_ssim += m["ssim"]

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{n} | valid so far: {valid}")

        if valid == 0:
            print("[evaluator] No valid images found (YOLO detected no persons).")
            return {}

        summary = {
            "valid_images": valid,
            "suppression_rate": suppressed / valid,
            "mean_conf_drop": total_conf_drop / valid,
            "mean_psnr_db": total_psnr / valid,
            "mean_ssim": total_ssim / valid,
        }

        # format output message
        output_lines = [
            "\n" + "=" * 50,
            "EVALUATION SUMMARY",
            "=" * 50,
            f"  Valid images (orig. had person): {summary['valid_images']}",
            f"  Suppression rate: {summary['suppression_rate']*100:.1f}%",
            f"  Mean confidence drop: {summary['mean_conf_drop']:.4f}",
            f"  Mean PSNR (dB): {summary['mean_psnr_db']:.2f}  (>35 = imperceptible)",
            f"  Mean SSIM: {summary['mean_ssim']:.4f}  (1.0 = identical)",
            "=" * 50 + "\n",
        ]
        
        # print to stdout
        for line in output_lines:
            print(line)
        
        # save to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.cfg.sample_dir / f"eval_summary_{timestamp}.txt"
        with open(output_file, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        
        print(f"[evaluator] Results saved to -> {output_file}")

        return summary