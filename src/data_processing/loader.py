"""
Data loader module for loading and preparing datasets for fine-tuning.
"""

from datasets import load_dataset, DatasetDict
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Handles loading and initial preparation of datasets from various sources.
    """
    
    def __init__(self, dataset_name: str, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            cache_dir: Optional directory to cache the dataset
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.dataset = None
        
    def load_dataset(self, split: Optional[str] = None) -> DatasetDict:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            split: Optional specific split to load ('train', 'test', etc.)
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            if split:
                self.dataset = load_dataset(
                    self.dataset_name,
                    split=split,
                    cache_dir=self.cache_dir
                )
            else:
                self.dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=self.cache_dir
                )
            
            logger.info(f"Dataset loaded successfully. Size: {len(self.dataset)}")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        info = {
            "name": self.dataset_name,
            "num_examples": len(self.dataset),
            "features": self.dataset.features,
            "splits": list(self.dataset.keys()) if isinstance(self.dataset, DatasetDict) else None
        }
        
        return info
    
    def split_dataset(
        self,
        train_size: float = 0.85,
        val_size: float = 0.10,
        test_size: float = 0.05,
        seed: int = 42
    ) -> DatasetDict:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            test_size: Proportion of data for testing
            seed: Random seed for reproducibility
            
        Returns:
            DatasetDict with train, validation, and test splits
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split sizes must sum to 1.0"
        
        logger.info("Splitting dataset...")
        
        # First split: train vs (val + test)
        train_val_test = self.dataset.train_test_split(
            test_size=(val_size + test_size),
            seed=seed
        )
        
        # Second split: val vs test
        val_test_split = val_size / (val_size + test_size)
        val_test = train_val_test['test'].train_test_split(
            test_size=(1 - val_test_split),
            seed=seed
        )
        
        dataset_dict = DatasetDict({
            'train': train_val_test['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        })
        
        logger.info(f"Dataset split - Train: {len(dataset_dict['train'])}, "
                   f"Val: {len(dataset_dict['validation'])}, "
                   f"Test: {len(dataset_dict['test'])}")
        
        self.dataset = dataset_dict
        return dataset_dict


def load_medical_dataset(
    dataset_name: str = "medalpaca/medical_meadow_medical_flashcards",
    cache_dir: Optional[str] = None
) -> DatasetLoader:
    """
    Convenience function to load medical dataset.
    
    Args:
        dataset_name: Name of the medical dataset
        cache_dir: Optional cache directory
        
    Returns:
        DatasetLoader instance with loaded dataset
    """
    loader = DatasetLoader(dataset_name, cache_dir)
    loader.load_dataset()
    return loader


if __name__ == "__main__":
    # Example usage
    loader = load_medical_dataset()
    print(loader.get_dataset_info())
    
    # Split the dataset
    dataset_dict = loader.split_dataset()
    print(f"\nSplit dataset:")
    for split, data in dataset_dict.items():
        print(f"{split}: {len(data)} examples")
