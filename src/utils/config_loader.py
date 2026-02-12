"""
Configuration loader utility for loading YAML configuration files.
"""

import yaml
from typing import Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Handles loading and accessing configuration from YAML files.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = None
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
        """
        logger.info(f"Loading configuration from: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info("Configuration loaded successfully")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Dictionary containing section configuration
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        return self.config.get(section, {})
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise ValueError("No configuration to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to: {output_path}")


def load_training_config(config_path: str = "configs/training_config.yaml") -> ConfigLoader:
    """
    Convenience function to load training configuration.
    
    Args:
        config_path: Path to training config file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)


if __name__ == "__main__":
    # Example usage
    try:
        config = load_training_config()
        
        # Access values
        model_name = config.get("model.name")
        learning_rate = config.get("training.learning_rate")
        
        print(f"Model: {model_name}")
        print(f"Learning Rate: {learning_rate}")
        
        # Get entire section
        lora_config = config.get_section("lora")
        print(f"\nLoRA Config: {lora_config}")
        
    except FileNotFoundError as e:
        print(f"Config file not found: {e}")
