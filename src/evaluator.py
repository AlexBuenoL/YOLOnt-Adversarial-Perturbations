from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Callable
from tqdm import tqdm

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from config import Config, cfg
from dataset import build_stream
from model import build_perturbation_net

logger = logging.getLogger(__name__)


# Helpers

def psnr(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """
    Peak signal-to-Noise ratio in dB (higher = less distortion).
    
    Args:
        original: The original image tensor.
        adversarial: The adversarial image tensor.
        
    Returns:
        PSNR value in decibels.
    """
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

    Args:
        original: The original image tensor.
        adversarial: The adversarial image tensor.
        window_size: The size of the Gaussian window for local statistics.
        C1: Stability constant to avoid division by zero.
        C2: Stability constant to avoid division by zero.

    Returns:
        SSIM value (float).
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


def _max_confidence_for_class(results, class_id: int) -> float:
    """
    Extract maximum confidence for a specific class from YOLO results.
    
    Args:
        results: YOLO result object for a single image.
        class_id: The class ID to check confidence for (e.g. 0 for person in COCO).
    
    Returns:
        The maximum confidence score for the specified class, or 0.0 if not detected.
    """
    if results.boxes is None or len(results.boxes) == 0:
        return 0.0
    
    class_boxes = results.boxes[results.boxes.cls == class_id]
    if len(class_boxes) == 0:
        return 0.0
    
    return float(class_boxes.conf.max().item())

def has_person_detection(results, class_id: int = 0, threshold: float = 0.25) -> bool:
    """
    Determines whether a YOLO result contains a valid detection for a given class.

    A detection is considered valid if at least one bounding box of the target class
    has a confidence score greater than or equal to the threshold.

    Args:
        results: YOLO prediction result for a single image.
        class_id: Target class ID (0 = person in COCO).
        threshold: Confidence threshold for valid detection.

    Returns:
        True if at least one detection exists above threshold, else False.
    """

    if results.boxes is None or len(results.boxes) == 0:
        return False

    class_mask = results.boxes.cls == class_id
    class_boxes = results.boxes[class_mask]

    if len(class_boxes) == 0:
        return False

    return (class_boxes.conf >= threshold).any().item()

# -- Evaluator --

class Evaluator:
    """
    Load a trained perturbation network and evaluate it against YOLO.
    """
    def __init__(
        self,
        config: Config = cfg,
        checkpoint_path: str | Path | None = None,
    ):
        self.cfg = config
        self.device = torch.device(config.device)

        self.net = build_perturbation_net(config).to(self.device)

        if checkpoint_path is not None:
            ckpt = torch.load(Path(checkpoint_path), map_location=self.device)
            self.net.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
        else:
            logger.info("No checkpoint provided; using random weights.")

        self.net.eval()

        self.yolo = YOLO(config.yolo_weights)

        self.stream = build_stream(config, split_type="eval")

    @torch.no_grad()
    def _evaluate_single(
        self, orig_tensor: torch.Tensor, transformation: Callable = None
    ) -> dict:
        """
        Evaluate one image.  
        Returns a dict of metrics per image.

        Args:
            orig_tensor: The original image tensor (C, H, W) in [0, 1].
        
        Returns:
            dict: Dictionary containing metrics such as: psnr, ssim, original and adversarial confidence scores, etc.
        """
        adv_tensor, _ = self.net(orig_tensor)

        # Apply optional post-perturbation transformation to adversarial tensor
        if transformation:
            adv_tensor = transformation(adv_tensor)

        # Convert tensors to numpy uint8 for YOLO inference
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

        orig_conf = _max_confidence_for_class(orig_results[0], class_id=0)
        adv_conf = _max_confidence_for_class(adv_results[0], class_id=0)

        return {
            "orig_has_person": has_person_detection(orig_results[0], class_id=0, threshold=0.25),
            "adv_has_person": has_person_detection(adv_results[0], class_id=0, threshold=0.25),
            "orig_conf": orig_conf,
            "adv_conf": adv_conf,
            "conf_drop": orig_conf - adv_conf,
            "psnr": psnr(orig_tensor, adv_tensor),
            "ssim": ssim(orig_tensor, adv_tensor),
        }

    def run(self, steps: int | None = None, transformation: Callable = None) -> dict:
        """
        Run evaluation over `steps` images.

        Args:
            steps: Number of images to evaluate. If None, uses config.eval_steps.

        Returns:
            dict: Summary of evaluation metrics across all processed images.
        """
        n = steps or self.cfg.eval_steps

        logger.info(f"Processing {n} images from eval partition...")

        # Only count images where YOLO detects a person
        valid = 0
        suppressed = 0
        total_conf_drop = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        for i in tqdm(range(n), desc="Evaluating images"):
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
                logger.info(f"Processed {i+1}/{n} | valid so far: {valid}")

        if valid == 0:
            logger.warning("No valid images found (YOLO detected no persons).")
            return {}

        summary = {
            "valid_images": valid,
            "suppression_rate": suppressed / valid,
            "mean_conf_drop": total_conf_drop / valid,
            "mean_psnr_db": total_psnr / valid,
            "mean_ssim": total_ssim / valid,
        }

        # Log results in a nice format
        output_lines = [
            "=" * 50,
            f"  Valid images (orig. had person): {summary['valid_images']}",
            f"  Suppression rate: {summary['suppression_rate']*100:.1f}%",
            f"  Mean confidence drop: {summary['mean_conf_drop']:.4f}",
            f"  Mean PSNR (dB): {summary['mean_psnr_db']:.2f}  (>35 = imperceptible)",
            f"  Mean SSIM: {summary['mean_ssim']:.4f}  (1.0 = identical)",
            "=" * 50 + "\n",
        ]
        for line in output_lines:
            logger.info(line)
        
        # Save results to a file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.cfg.sample_dir / f"eval_summary_{timestamp}.txt"
        with open(output_file, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        
        logger.info(f"Results saved to {output_file}")

        return summary