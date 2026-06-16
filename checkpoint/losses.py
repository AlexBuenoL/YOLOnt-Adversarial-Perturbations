from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from config import Config, cfg


def detection_loss(
    raw_yolo_output: torch.Tensor,
    target_class_id: int,
    topk: int,
) -> torch.Tensor:
    """
    Reduces person detections by minimizing the top-k class confidence scores.

    Args:
        raw_yolo_output: The raw output tensor from YOLO.
        target_class_id: The class ID to target (e.g., 0 for person in COCO).
        topk: The number of top detection scores to consider for the loss.

    Returns:
        A scalar tensor representing the detection loss.
    """
    person_scores = raw_yolo_output[:, 4 + target_class_id, :]

    # Take the mean of the top-k highest person confidence scores as the loss.
    # TODO: We could also experiment with sum or maximum instead of mean.
    k = min(topk, person_scores.numel())
    topk_scores = torch.topk(person_scores.flatten(), k=k).values

    return topk_scores.mean()

def reconstruction_loss(
    adv_image: torch.Tensor,
    orig_image: torch.Tensor,
) -> torch.Tensor:
    """
    Pixel-wise MSE between the adversarial image and the original.

    Args:
        adv_image: The adversarially perturbed image tensor.
        orig_image: The original image tensor.
    
    Returns:
        A scalar tensor representing the reconstruction loss.
    """
    return F.mse_loss(adv_image, orig_image)

def total_variation_loss(perturbation: torch.Tensor) -> torch.Tensor:
    """
    Total variation of the perturbation tensor

    Args:
        perturbation: The perturbation tensor output by the UNet (B, 3, H, W)
    
    Returns:
        A scalar tensor representing the total variation loss.
    """
    diff_h = torch.abs(perturbation[:, :, :-1, :] - perturbation[:, :, 1:, :])
    diff_w = torch.abs(perturbation[:, :, :, :-1] - perturbation[:, :, :, 1:])
    return diff_h.mean() + diff_w.mean()



# YOLO input preprocessing

def preprocess_for_yolo(
    image_tensor: torch.Tensor,
    yolo_size: int,
) -> torch.Tensor:
    """
    Resize a (1, 3, H, W) tensor to the size YOLO expects.
    Kept differentiable so gradients flow back through the resize.
    """
    return F.interpolate(
        image_tensor,
        size=(yolo_size, yolo_size),
        mode="bilinear",
        align_corners=False,
    )


# Combined loss class

class AdversarialLoss:
    """
    Aggregates the three loss components.

    Supports adaptive lambda scheduling.
    """

    def __init__(self, config: Config = cfg) -> None:
        self.cfg = config

    def _compute_adaptive_lambdas(self, l_det: float) -> tuple[float, float]:
        """
        Compute adaptive lambda weights based on current detection loss.
        
        When det_loss < threshold, increase lambda_recon and lambda_tv
        to focus on reconstruction quality instead of more suppresion.
        """
        if not self.cfg.use_adaptive_lambdas or l_det >= self.cfg.det_loss_threshold:
            return self.cfg.lambda_recon, self.cfg.lambda_tv
        
        progress = 1.0 - (l_det / self.cfg.det_loss_threshold)
        progress = min(1.0, max(0.0, progress))
        
        progress = progress ** 2
        
        lambda_recon_adaptive = self.cfg.lambda_recon + (self.cfg.lambda_recon_max - self.cfg.lambda_recon) * progress
        lambda_tv_adaptive = self.cfg.lambda_tv + (self.cfg.lambda_tv_max - self.cfg.lambda_tv) * progress
        
        return lambda_recon_adaptive, lambda_tv_adaptive

    def __call__(
        self,
        raw_yolo_output: torch.Tensor,
        adv_image: torch.Tensor,
        orig_image: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute and return the total loss plus a loggable info.

        Args:
            raw_yolo_output: The raw YOLO output tensor.
            adv_image: The adversarially perturbed image tensor.
            orig_image: The original image tensor.
            perturbation: The perturbation tensor output by the UNet.

        Returns:
            A tuple containing the total loss and a dictionary of loss components.
        """
        l_det = detection_loss(
            raw_yolo_output,
            self.cfg.target_class_id,
            self.cfg.topk_detections,
        )
        l_recon = reconstruction_loss(adv_image, orig_image)
        l_tv    = total_variation_loss(perturbation)
        
        lambda_recon_adaptive, lambda_tv_adaptive = self._compute_adaptive_lambdas(l_det.item())

        total = (
            l_det
            + lambda_recon_adaptive * l_recon
            + lambda_tv_adaptive * l_tv
        )

        breakdown = {
            "det": l_det.item(),
            "recon": l_recon.item(),
            "tv": l_tv.item(),
            "total": total.item(),
            "lambda_recon": lambda_recon_adaptive,
            "lambda_tv": lambda_tv_adaptive,
        }

        return total, breakdown