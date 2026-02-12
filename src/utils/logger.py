"""
Logging utility for consistent logging across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file name (timestamp will be added)
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{log_file}_{timestamp}.log"
        file_path = log_path / file_name
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    return logger


class ExperimentLogger:
    """
    Logger for tracking experiment metrics and events.
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        self.logger = setup_logger(
            name=f"experiment.{experiment_name}",
            log_file=experiment_name,
            log_dir=log_dir
        )
        
        self.metrics = {}
        
    def log_hyperparameters(self, hyperparameters: dict):
        """
        Log hyperparameters for the experiment.
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        self.logger.info("=" * 50)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("=" * 50)
        self.logger.info("Hyperparameters:")
        
        for key, value in hyperparameters.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append((step, value))
        
        if step is not None:
            self.logger.info(f"Step {step} - {name}: {value:.4f}")
        else:
            self.logger.info(f"{name}: {value:.4f}")
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(name, value, step)
    
    def log_training_start(self):
        """Log the start of training."""
        self.logger.info("=" * 50)
        self.logger.info("Training Started")
        self.logger.info("=" * 50)
    
    def log_training_end(self, duration_seconds: float):
        """
        Log the end of training.
        
        Args:
            duration_seconds: Training duration in seconds
        """
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        
        self.logger.info("=" * 50)
        self.logger.info("Training Completed")
        self.logger.info(f"Total Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        self.logger.info("=" * 50)
    
    def log_evaluation_results(self, results: dict):
        """
        Log evaluation results.
        
        Args:
            results: Dictionary of evaluation metrics
        """
        self.logger.info("=" * 50)
        self.logger.info("Evaluation Results")
        self.logger.info("=" * 50)
        
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")


if __name__ == "__main__":
    # Example usage
    
    # Basic logger
    logger = setup_logger("test_logger", log_file="test")
    logger.info("This is a test log message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Experiment logger
    exp_logger = ExperimentLogger("experiment_1")
    
    exp_logger.log_hyperparameters({
        "learning_rate": 2e-4,
        "batch_size": 8,
        "epochs": 3
    })
    
    exp_logger.log_training_start()
    
    for epoch in range(3):
        exp_logger.log_metric("train_loss", 0.5 - epoch * 0.1, step=epoch)
        exp_logger.log_metric("val_loss", 0.6 - epoch * 0.1, step=epoch)
    
    exp_logger.log_training_end(3600)
    
    exp_logger.log_evaluation_results({
        "bleu_score": 0.45,
        "rouge_l": 0.52,
        "perplexity": 15.3
    })
