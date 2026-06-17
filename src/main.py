from __future__ import annotations

import argparse
import logging

from config import Config
from logging_config import setup_logging
from trainer import Trainer
from evaluator import Evaluator

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial Perturbation Network")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="both",
        help="Mode of operation: 'train', 'eval', or 'both'.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training or for evaluation.",
    )
    return parser.parse_args()

def main() -> None:

    setup_logging()
    args = parse_args()
    config = Config()

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