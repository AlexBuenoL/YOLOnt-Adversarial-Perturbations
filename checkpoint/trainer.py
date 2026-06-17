from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from config import Config, cfg
from dataset import build_stream
from losses import AdversarialLoss, preprocess_for_yolo
from model import PerturbationUNet, build_perturbation_net

logger = logging.getLogger(__name__)


# Helper functions

def _load_yolo(weights: str, device: torch.device) -> torch.nn.Module:
    """
    Load YOLOv8, freeze all parameters, and set it to eval mode.

    Args:
        weights: Path to YOLO weights file (e.g. `yolov8n.pt`).
        device: torch device to load the model on.

    Returns:
        The loaded YOLO model with frozen parameters.
    """
    yolo = YOLO(weights)
    model = yolo.model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"YOLO backbone loaded | params: {total_params:,} (frozen)")
    return model


def _save_checkpoint(
    net: PerturbationUNet,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: Path,
) -> None:
    """
    Saves model and optimizer state dicts along with epoch and step info for resuming training later.

    Args:
        net: The perturbation network to save.
        optimizer: The optimizer whose state to save.
        epoch: Current epoch number (for checkpoint naming).
        step: Current global step number (for checkpoint naming).
        checkpoint_dir: Directory where checkpoints should be saved.
    """
    path = checkpoint_dir / f"ckpt_ep{epoch:03d}_s{step:06d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint saved in {path}")


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

# Visualization helpers

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert (3, H, W) torch tensor [0,1] to PIL Image.
    
    Args:
        tensor: A torch.Tensor of shape (3, H, W) with pixel values in [0, 1].
    
    Returns:
        out: A PIL.Image object.
    """
    img_np = (tensor.cpu().detach().permute(1, 2, 0).numpy() * 255).astype('uint8')
    return Image.fromarray(img_np)

def _draw_prediction_text(
    pil_img: Image.Image, 
    confidence: float,
    position: str = "top"
) -> Image.Image:
    """
    Draw YOLO prediction text on image.

    Args:
        pil_img: PIL Image to draw on.
        confidence: Confidence score to display.
        position: "top" or "bottom" to indicate where to place the text.

    Returns:
        PIL Image with drawn text.
    """
    draw = ImageDraw.Draw(pil_img)
    text = f"Conf: {confidence:.3f}"
    
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position text
    if position == "top":
        x, y = 5, 5
    else:
        x, y = 5, pil_img.height - text_height - 5
    
    # Draw background rectangle
    padding = 2
    draw.rectangle(
        [(x - padding, y - padding), 
         (x + text_width + padding, y + text_height + padding)],
        fill=(0, 0, 0)
    )
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return pil_img

def _save_samples(
    net: PerturbationUNet,
    sample_images: list[torch.Tensor],
    yolo: YOLO,
    step: int,
    sample_dir: Path,
) -> None:
    """
    Run the network on a fixed set of sample images and save them in a grid.
    Show original image, adversarial image, and perturbation (normalized for visibility) side by side.
    
    Also displays YOLO person detection confidence on original and adversarial.

    Args:
        net: The perturbation network to generate adversarial images.
        sample_images: A list of original images to use as samples (as torch tensors).
        yolo: A YOLO model for inference to get confidence scores.
        step: Current global step number (for naming the sample image).
        sample_dir: Directory where sample images should be saved.
    """
    net.eval()
    grid_images = []

    with torch.no_grad():
        for img in sample_images:
            adv, perturb = net(img)
            
            # Convert tensors to numpy uint8 for YOLO inference
            orig_np = img.squeeze(0).cpu().permute(1, 2, 0).mul(255).byte().numpy()
            adv_np = adv.squeeze(0).cpu().permute(1, 2, 0).mul(255).byte().numpy()
            
            # Get YOLO predictions
            orig_results = yolo(orig_np, verbose=False)
            adv_results = yolo(adv_np, verbose=False)
            
            orig_conf = _max_confidence_for_class(orig_results[0], class_id=0)
            adv_conf = _max_confidence_for_class(adv_results[0], class_id=0)
            
            # Convert to PIL and add prediction text
            orig_pil = _tensor_to_pil(img[0])
            adv_pil = _tensor_to_pil(adv[0])
            
            orig_pil = _draw_prediction_text(orig_pil, orig_conf, position="top")
            adv_pil = _draw_prediction_text(adv_pil, adv_conf, position="top")
            
            # Convert back to tensors
            orig_tensor = TF.to_tensor(orig_pil)
            adv_tensor = TF.to_tensor(adv_pil)
            
            # Normalize perturbation between [0,1]
            p = perturb[0]
            p_vis = torch.zeros_like(p)
            for c in range(p.shape[0]):
                p_c = p[c]
                p_min, p_max = p_c.min(), p_c.max()
                p_vis[c] = (p_c - p_min) / (p_max - p_min + 1e-8)
            
            grid_images.extend([orig_tensor, adv_tensor, p_vis])

    grid = vutils.make_grid(grid_images, nrow=3, padding=2, normalize=False)
    path = sample_dir / f"sample_s{step:06d}.png"
    vutils.save_image(grid, path)
    logger.info(f"Sample grid saved in {path}")

    net.train()


# Trainer class

class Trainer:

    def __init__(
        self,
        config: Config = cfg,
        resume_from: str | Path | None = None,
    ):
    
        self.cfg = config
        self.device = torch.device(config.device)
        self.net = build_perturbation_net(config).to(self.device)
        self.yolo = _load_yolo(config.yolo_weights, self.device)
        self.yolo_wrapper = YOLO(config.yolo_weights)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.criterion = AdversarialLoss(config)
        self.start_epoch = 0
        self.global_step = 0

        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

        self.stream = build_stream(config, split_type="train")
        logger.info(f"Caching {config.num_sample_images} sample images with person detections...")
        self.sample_images: list[torch.Tensor] = []
        
        attempts = 0
        max_attempts = config.num_sample_images * 20 # Safety net to avoid infinite loops
        
        while len(self.sample_images) < config.num_sample_images and attempts < max_attempts:
            img = next(self.stream).to(self.device)

            img_np = img.squeeze(0).cpu().permute(1, 2, 0).mul(255).byte().numpy()
            results = self.yolo_wrapper(img_np, verbose=False)
            
            if _max_confidence_for_class(results[0], class_id=0) > 0.0:
                self.sample_images.append(img)
                logger.debug(f"Found image with person detection [{len(self.sample_images)}/{config.num_sample_images}]")
            
            attempts += 1
        
        if len(self.sample_images) < config.num_sample_images:
            logger.warning(f"Only found {len(self.sample_images)}/{config.num_sample_images} images with person detections after {attempts} attempts")

    def _load_checkpoint(self, path: Path) -> None:
        """
        Load model and optimizer state dicts from a checkpoint file, along with epoch and step info.
        
        Args:
            path: Path to the checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt["epoch"]
        self.global_step = ckpt["step"]
        logger.info(f"Checkpoint loaded from {path} (epoch {self.start_epoch}, step {self.global_step})")

    def _train_step(self, orig_image: torch.Tensor) -> dict[str, float]:
        """
        Single optimization step.

        Args:
            orig_image: A batch of original images (B, 3, H, W)
        
        Returns:
            A dictionary with loss breakdown for logging.
        """
        orig_image = orig_image.to(self.device)

        self.optimizer.zero_grad()

        # 1. Generate adversarial image and perturbation
        adv_image, perturbation = self.net(orig_image)

        # 2: Preprocess adversarial image for YOLO input
        adv_yolo = preprocess_for_yolo(adv_image, self.cfg.yolo_input_size)

        # 3: YOLO forward pass (returns raw tensor at index 0)
        with torch.no_grad():
            raw_tensor = self.yolo(adv_yolo)[0]

        # 4: Compute losses
        total_loss, breakdown = self.criterion(
            raw_yolo_output=raw_tensor,
            adv_image=adv_image,
            orig_image=orig_image,
            perturbation=perturbation,
        )

        # 5: Backpropagation and optimization step
        total_loss.backward()
        self.optimizer.step()

        return breakdown

    def train(self) -> None:
        """Run the full training."""
        logger.info(
            f"Starting training | epochs={self.cfg.epochs} | steps/epoch={self.cfg.steps_per_epoch}"
        )

        self.net.train()

        for epoch in range(self.start_epoch, int(self.cfg.epochs)):
            running = {"det": 0.0, "recon": 0.0, "tv": 0.0, "total": 0.0}
            count = 0

            for _ in range(self.cfg.steps_per_epoch):
                self.global_step += 1
                count += 1

                image = next(self.stream).to(self.device)
                breakdown = self._train_step(image)

                for k in running:
                    running[k] += breakdown[k]

                # logging
                if self.global_step % self.cfg.log_every == 0:
                    avg = {k: running[k] / count for k in running}
                    lambda_info = ""
                    if self.cfg.use_adaptive_lambdas and 'lambda_recon' in avg:
                        lambda_info = f" [lambda_r={avg['lambda_recon']:.1f} lambda_t={avg['lambda_tv']:.3f}]"
                    logger.info(
                        f"Epoch {epoch+1:02d}/{self.cfg.epochs} | "
                        f"Step {self.global_step:06d} | "
                        f"Loss {avg['total']:.4f} "
                        f"(det={avg['det']:.4f} "
                        f"recon={avg['recon']:.6f} "
                        f"tv={avg['tv']:.6f}){lambda_info}"
                    )
                    # Reset running totals after logging
                    running = {k: 0.0 for k in running}
                    count = 0

                # Checkpointing
                if self.global_step % self.cfg.save_every == 0:
                    _save_checkpoint(
                        self.net,
                        self.optimizer,
                        epoch,
                        self.global_step,
                        self.cfg.checkpoint_dir,
                    )

                # Sample generation
                if self.global_step % self.cfg.sample_every == 0:
                    _save_samples(
                        self.net,
                        self.sample_images,
                        self.yolo_wrapper,
                        self.global_step,
                        self.cfg.sample_dir,
                    )

        # Final checkpoint after training completes
        _save_checkpoint(
            self.net,
            self.optimizer,
            self.cfg.epochs,
            self.global_step,
            self.cfg.checkpoint_dir,
        )
        logger.info("Training complete.")