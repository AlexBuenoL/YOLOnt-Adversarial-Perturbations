from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    
    # Paths
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    sample_dir: Path = Path("outputs/samples")

    # Device
    device: str = "cpu"           
                                    
    # Dataset
    hf_dataset_name: str = "bitmind/MS-COCO-unique-256_training_faces"
    hf_config_name: str = "base_transforms"
    hf_split: str = "train"
    hf_train_split_ratio: float = 0.8
    dataset_size: int = 7180     
    image_size: int = 256         

    # YOLO
    yolo_weights: str = "yolov8n.pt"
    yolo_input_size: int = 640 # YOLO expects 640x640
    target_class_id: int = 0 # COCO class 0 = person

    # Top-k detection scores to suppress per image
    topk_detections: int = 20

    # UNet architecture
    unet_base_channels: int = 8

    # Maximum perturbation magnitude
    epsilon: float = 8 / 255.0

    # Lambdas for loss components
    lambda_recon: float = 50.0 
    lambda_tv: float = 5
    
    # Adaptative lambda scheduling parameters
    use_adaptive_lambdas: bool = True
    det_loss_threshold: float = 0.05
    lambda_recon_max: float = 250.0
    lambda_tv_max: float = 25.0

    # Training
    epochs: int = 1               
    steps_per_epoch: int | None = None  
    learning_rate: float = 1e-3
    log_every: int = 50           
    save_every: int = 250         
    sample_every: int = 250        
    num_sample_images: int = 4     

    # Evaluation
    eval_steps: int | None = None 

    def __post_init__(self):
        # Create necessary directories
        for d in (self.output_dir, self.checkpoint_dir, self.sample_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        
        # Compute steps_per_epoch if not set, based on dataset size and train split ratio
        if self.steps_per_epoch is None:
            train_size = int(self.dataset_size * self.hf_train_split_ratio)
            self.steps_per_epoch = train_size // self.epochs
        
        # Compute eval_steps if not set, based on dataset size and eval split ratio
        if self.eval_steps is None:
            eval_partition_ratio = 1.0 - self.hf_train_split_ratio
            self.eval_steps = int(self.dataset_size * eval_partition_ratio)

cfg = Config()