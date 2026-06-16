import logging
from pathlib import Path


def setup_logging(log_dir: Path | str = "logs", log_level: int = logging.INFO) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        log_dir: Directory where log files will be saved.
        log_level: Logging level (default: logging.INFO).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)-20s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        fmt="%(name)-15s [%(levelname)-7s] %(message)s"
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (more detailed)
    log_file = log_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured | log_dir={log_dir} | level={logging.getLevelName(log_level)}")
