# Adversarial Textures — Anti-YOLO

A UNet-based adversarial perturbation network that learns to generate imperceptible texture perturbations to suppress YOLOv8 person detections.

## Setup

```bash
pip install -r requirements.txt
```

The dataset is streamed from HuggingFace (`bitmind/MS-COCO-unique-256_training_faces`), no manual download needed. The YOLOv8 weights (`yolov8n.pt`) are included in the repo.

### Train + Evaluate

```bash
python src/main.py
```

### Train only

```bash
python main.py --mode train
```

### Evaluate from a checkpoint

```bash
python main.py --mode eval --resume outputs/checkpoints/ckpt_ep000_s004000.pt
```

### Test robustness to JPEG compression

```bash
python test_compression.py --resume outputs/checkpoints/ckpt_ep000_s004000.pt
```

### Test robustness to black & white conversion

```bash
python test_black_white.py --resume outputs/checkpoints/ckpt_ep000_s004000.pt
```

## Outputs

| Path | Contents |
|---|---|
| `outputs/checkpoints/` | Saved model checkpoints |
| `outputs/samples/` | Sample perturbation images saved during training |
| `outputs/metrics/` | Evaluation metrics |

## Key Config

Edit `src/config.py` to change hyperparameters (epsilon, learning rate, epochs, UNet channels, etc.).
