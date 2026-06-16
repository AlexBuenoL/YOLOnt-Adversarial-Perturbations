from __future__ import annotations

import argparse
import logging

from config import Config
from logging_config import setup_logging
from trainer import Trainer
from evaluator import Evaluator

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adversarial perturbation network against YOLOv8."
    )

    p.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="train",
        help="'train', 'eval', or 'both' (train then eval).",
    )

    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a .pt checkpoint to resume training or load for eval.",
    )

    # Allow quick overrides without editing config.py for convenience
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps_per_epoch", type=int, default=None)
    p.add_argument("--lr",  type=float, default=None)
    p.add_argument("--epsilon", type=float, default=None, help="Max perturbation magnitude (default: 8/255).")
    p.add_argument("--lambda_recon", type=float, default=None)
    p.add_argument("--lambda_tv", type=float, default=None)
    p.add_argument("--no_adaptive_lambdas", action="store_true", default=False, help="Disable adaptive lambda scheduling.")
    p.add_argument("--det_loss_threshold", type=float, default=None, help="Detection loss threshold for adaptive scheduling (default: 0.1).")
    p.add_argument("--lambda_recon_max", type=float, default=None, help="Max reconstruction weight when det_loss is very low (default: 500).")
    p.add_argument("--lambda_tv_max", type=float, default=None, help="Max TV weight when det_loss is very low (default: 1.0).")
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--topk", type=int, default=None, help="Top-k detection scores to suppress per image.")
    p.add_argument("--base_channels", type=int, default=None, help="UNet base channel count (default: 8).")
    p.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name override.")
    p.add_argument("--hf_config", type=str, default=None, help="HuggingFace config name: 'base_transforms' or 'random_aug_transforms'.")
    p.add_argument("--hf_train_split_ratio", type=float, default=None, help="Train/eval split ratio (default: 0.8 = 80% train, 20% eval).")
    p.add_argument("--dataset_size", type=int, default=None, help="Total dataset size (default: 7180). Used to calculate eval_steps.")
    p.add_argument("--device", type=str, default=None, help="Device to use: 'cpu', 'cuda', 'cuda:0', etc. (default: 'cpu').")

    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    # Build config and override with any command-line args provided
    config = Config()
    if args.epochs is not None: config.epochs = args.epochs
    if args.steps_per_epoch is not None: config.steps_per_epoch  = args.steps_per_epoch
    if args.lr is not None: config.learning_rate = args.lr
    if args.epsilon is not None: config.epsilon = args.epsilon
    if args.lambda_recon is not None: config.lambda_recon = args.lambda_recon
    if args.lambda_tv is not None: config.lambda_tv = args.lambda_tv
    if args.no_adaptive_lambdas: config.use_adaptive_lambdas = False
    if args.det_loss_threshold is not None: config.det_loss_threshold = args.det_loss_threshold
    if args.lambda_recon_max is not None: config.lambda_recon_max = args.lambda_recon_max
    if args.lambda_tv_max is not None: config.lambda_tv_max = args.lambda_tv_max
    if args.eval_steps is not None: config.eval_steps = args.eval_steps
    if args.topk is not None: config.topk_detections = args.topk
    if args.base_channels is not None: config.unet_base_channels = args.base_channels
    if args.dataset is not None: config.hf_dataset_name = args.dataset
    if args.hf_config is not None: config.hf_config_name = args.hf_config
    if args.hf_train_split_ratio is not None: config.hf_train_split_ratio = args.hf_train_split_ratio
    if args.dataset_size is not None: config.dataset_size = args.dataset_size
    if args.device is not None: config.device = args.device
    
    # In case of override, recalculate steps_per_epoch and eval_steps
    if args.dataset_size is not None:
        train_size = int(config.dataset_size * config.hf_train_split_ratio)
        config.steps_per_epoch = train_size // config.epochs
        eval_partition_ratio = 1.0 - config.hf_train_split_ratio
        config.eval_steps = int(config.dataset_size * eval_partition_ratio)

    logger.info("=" * 60)
    logger.info("    Adversarial Perturbation Network")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Dataset: {config.hf_dataset_name}")
    logger.info(f"HF Config: {config.hf_config_name}")
    logger.info(f"Total dataset: {config.dataset_size:,} images")
    logger.info(f"Data split: {config.hf_train_split_ratio:.1%} train ({int(config.dataset_size*config.hf_train_split_ratio):,}) / {1-config.hf_train_split_ratio:.1%} eval ({int(config.dataset_size*(1-config.hf_train_split_ratio)):,})")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Steps/epoch: {config.steps_per_epoch:,}")
    logger.info(f"Total train steps: {config.epochs * config.steps_per_epoch:,}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epsilon: {config.epsilon:.5f}  ({config.epsilon*255:.1f}/255)")
    logger.info(f"lambda_recon: {config.lambda_recon}")
    logger.info(f"lambda_tv: {config.lambda_tv}")
    if config.use_adaptive_lambdas:
        logger.info(f"Adaptive lambdas: enabled (threshold: {config.det_loss_threshold})")
        logger.info(f"  lambda_recon_max {config.lambda_recon_max}")
        logger.info(f"  lambda_tv_max: {config.lambda_tv_max}")
    logger.info(f"UNet base_ch: {config.unet_base_channels}")
    logger.info("=" * 60)

    if args.mode in ("train", "both"):
        trainer = Trainer(config=config, resume_from=args.resume)
        trainer.train()

        # After training, use the latest checkpoint for evaluation
        latest_ckpt = sorted(config.checkpoint_dir.glob("*.pt"))[-1]
        resume_for_eval = latest_ckpt
    else:
        resume_for_eval = args.resume

    if args.mode in ("eval", "both"):
        evaluator = Evaluator(config=config, checkpoint_path=resume_for_eval)
        evaluator.run()

if __name__ == "__main__":
    main()