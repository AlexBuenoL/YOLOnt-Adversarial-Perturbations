import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import torchvision.transforms as transforms

from evaluator import Evaluator, _max_confidence_for_class, psnr, ssim
from config import Config, cfg
from logging_config import setup_logging
from trainer import _draw_prediction_text

logger = logging.getLogger(__name__)

class BWEvaluator(Evaluator):
    """
    Subclasses the standard Evaluator to strictly test structural perturbations 
    on Black & White images, avoiding color-domain noise artifacts.
    """
    @torch.no_grad()
    def _evaluate_single_bw(self, orig_tensor: torch.Tensor, save_path: Path = None) -> dict:
        # 1. Convert input strictly to Grayscale (3 channels for YOLO compatibility)
        orig_bw = TF.rgb_to_grayscale(orig_tensor, num_output_channels=3)
        
        # 2. Get raw perturbation from model
        _, raw_perturbation = self.net(orig_bw)
        
        # 3. Force perturbation to be grayscale by averaging the RGB channels.
        # This prevents the final 1x1 conv layer from injecting colored noise.
        p_bw = raw_perturbation.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        
        # 4. Reconstruct the bounded adversarial image
        adv_bw = torch.clamp(orig_bw + p_bw, 0.0, 1.0)
        
        # 5. Convert tensors to numpy uint8 for YOLO inference
        def to_numpy(t):
            return t.squeeze(0).cpu().permute(1, 2, 0).mul(255).byte().numpy()

        orig_np = to_numpy(orig_bw)
        adv_np = to_numpy(adv_bw)

        # 6. Run YOLOv8 baseline and adversarial evaluations
        orig_results = self.yolo(orig_np, verbose=False)
        adv_results = self.yolo(adv_np, verbose=False)

        orig_conf = _max_confidence_for_class(orig_results[0], class_id=0)
        adv_conf = _max_confidence_for_class(adv_results[0], class_id=0)
        
        # Save visual proof if requested
        if save_path and orig_conf > 0.0:
            self._save_visual_proof(orig_bw, adv_bw, p_bw, orig_conf, adv_conf, save_path)

        return {
            "orig_has_person": orig_conf > 0.0,
            "adv_has_person": adv_conf > 0.0,
            "orig_conf": orig_conf,
            "adv_conf": adv_conf,
            "conf_drop": orig_conf - adv_conf,
            "psnr": psnr(orig_bw, adv_bw),
            "ssim": ssim(orig_bw, adv_bw),
        }

    def _save_visual_proof(
        self,
        orig,
        adv,
        perturb,
        orig_conf,
        adv_conf,
        save_path
    ):
        # Normalize perturbation for visualization
        p_min, p_max = perturb.min(), perturb.max()
        p_vis = (perturb - p_min) / (p_max - p_min + 1e-8)

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        # Convert images to PIL
        orig_pil = to_pil(orig.squeeze(0).cpu())
        adv_pil = to_pil(adv.squeeze(0).cpu())

        # Draw confidence values
        orig_pil = _draw_prediction_text(
            orig_pil,
            confidence=orig_conf,
            position="top"
        )

        adv_pil = _draw_prediction_text(
            adv_pil,
            confidence=adv_conf,
            position="top"
        )

        # Convert back to tensors
        orig_vis = to_tensor(orig_pil)
        adv_vis = to_tensor(adv_pil)

        # Grid: Original | Adversarial | Perturbation
        grid = vutils.make_grid(
            [
                orig_vis,
                adv_vis,
                p_vis.squeeze(0).cpu()
            ],
            nrow=3,
            padding=2
        )

        vutils.save_image(grid, save_path)

    def run_bw(self, steps: int | None = None, num_visuals: int = 10) -> dict:
        n = steps or self.cfg.eval_steps
        logger.info(f"Processing {n} images for BLACK & WHITE stress test...")

        valid, suppressed = 0, 0
        total_conf_drop, total_psnr, total_ssim = 0.0, 0.0, 0.0
        
        # Ensure visual output directory exists
        visuals_dir = self.cfg.sample_dir / "bw_proofs"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        saved_visuals = 0

        for i in range(n):
            img = next(self.stream).to(self.device)
            
            # Determine if we should save a visual proof for this iteration
            save_path = visuals_dir / f"bw_proof_{saved_visuals:03d}.png" if saved_visuals < num_visuals else None
            
            m = self._evaluate_single_bw(img, save_path=save_path)

            if not m["orig_has_person"]:
                continue

            valid += 1
            if save_path: 
                saved_visuals += 1

            if not m["adv_has_person"]:
                suppressed += 1

            total_conf_drop += m["conf_drop"]
            total_psnr += m["psnr"]
            total_ssim += m["ssim"]

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{n} | valid so far: {valid}")

        if valid == 0:
            logger.warning("No valid images found.")
            return {}

        summary = {
            "valid_images": valid,
            "suppression_rate": suppressed / valid,
            "mean_conf_drop": total_conf_drop / valid,
            "mean_psnr_db": total_psnr / valid,
            "mean_ssim": total_ssim / valid,
        }

        output_lines = [
            "=" * 50,
            "    BLACK & WHITE STRESS TEST SUMMARY",
            "=" * 50,
            f"Valid images (orig. had person): {summary['valid_images']}",
            f"Suppression rate: {summary['suppression_rate']*100:.1f}%",
            f"Mean confidence drop: {summary['mean_conf_drop']:.4f}",
            f"Mean PSNR (dB): {summary['mean_psnr_db']:.2f}",
            f"Mean SSIM: {summary['mean_ssim']:.4f}",
            "=" * 50,
        ]
        
        for line in output_lines:
            logger.info(line)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.cfg.sample_dir / f"bw_summary_{timestamp}.txt", "w") as f:
            for line in output_lines: f.write(line + "\n")

        return summary

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="B&W Stress Test for YOLOn't")
    parser.add_argument("--resume", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps", type=int, default=None, help="Number of images to evaluate")
    parser.add_argument("--visuals", type=int, default=15, help="Number of visual proofs to generate")
    args = parser.parse_args()

    config = Config()
    config.device = args.device
    config.eval_steps = args.eval_steps if args.eval_steps is not None else config.eval_steps

    logger.info(f"Loading checkpoint: {args.resume}")
    evaluator = BWEvaluator(config=config, checkpoint_path=args.resume)
    evaluator.run_bw(num_visuals=args.visuals)

if __name__ == "__main__":
    main()