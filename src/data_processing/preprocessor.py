"""
Data preprocessing module for cleaning and formatting data for model training.
"""

import re
from typing import Dict, List, Optional, Callable
from datasets import Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data cleaning, formatting, and preprocessing for LLM fine-tuning.
    """
    
    def __init__(
        self,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
        max_length: int = 512
    ):
        """
        Initialize the preprocessor.
        
        Args:
            instruction_template: Template for formatting instruction-response pairs
            max_length: Maximum sequence length
        """
        self.instruction_template = instruction_template
        self.max_length = max_length
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that may interfere with tokenization
        text = text.strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def format_instruction_response(
        self,
        instruction: str,
        response: str
    ) -> str:
        """
        Format instruction and response into training template.
        
        Args:
            instruction: The instruction/question
            response: The expected response/answer
            
        Returns:
            Formatted string
        """
        instruction = self.clean_text(instruction)
        response = self.clean_text(response)
        
        formatted = self.instruction_template.format(
            instruction=instruction,
            response=response
        )
        
        return formatted
    
    def preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single example.
        
        Args:
            example: Dictionary containing 'instruction' and 'response' keys
            
        Returns:
            Preprocessed example dictionary
        """
        # Handle different possible key names in the dataset
        instruction_keys = ['instruction', 'input', 'question', 'query']
        response_keys = ['response', 'output', 'answer', 'completion']
        
        instruction = None
        response = None
        
        for key in instruction_keys:
            if key in example:
                instruction = example[key]
                break
        
        for key in response_keys:
            if key in example:
                response = example[key]
                break
        
        if instruction is None or response is None:
            raise ValueError(f"Could not find instruction/response in example: {example.keys()}")
        
        formatted_text = self.format_instruction_response(instruction, response)
        
        return {
            "text": formatted_text,
            "instruction": instruction,
            "response": response
        }
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        num_proc: Optional[int] = None,
        remove_columns: Optional[List[str]] = None
    ) -> Dataset:
        """
        Preprocess an entire dataset.
        
        Args:
            dataset: Input dataset
            num_proc: Number of processes for parallel processing
            remove_columns: Columns to remove after preprocessing
            
        Returns:
            Preprocessed dataset
        """
        logger.info(f"Preprocessing dataset with {len(dataset)} examples...")
        
        processed_dataset = dataset.map(
            self.preprocess_example,
            num_proc=num_proc,
            desc="Preprocessing examples"
        )
        
        if remove_columns:
            processed_dataset = processed_dataset.remove_columns(remove_columns)
        
        logger.info("Preprocessing complete.")
        return processed_dataset
    
    def filter_by_length(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: Optional[int] = None
    ) -> Dataset:
        """
        Filter examples that exceed maximum length after tokenization.
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer to use for length checking
            max_length: Maximum sequence length (uses self.max_length if None)
            
        Returns:
            Filtered dataset
        """
        max_len = max_length or self.max_length
        
        def is_valid_length(example):
            tokens = tokenizer(example["text"], truncation=True, max_length=max_len)
            return len(tokens["input_ids"]) <= max_len
        
        logger.info(f"Filtering examples by length (max: {max_len})...")
        original_size = len(dataset)
        
        filtered_dataset = dataset.filter(
            is_valid_length,
            desc="Filtering by length"
        )
        
        filtered_size = len(filtered_dataset)
        logger.info(f"Filtered {original_size - filtered_size} examples. "
                   f"Remaining: {filtered_size}")
        
        return filtered_dataset
    
    def remove_duplicates(self, dataset: Dataset) -> Dataset:
        """
        Remove duplicate examples from dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with duplicates removed
        """
        logger.info("Removing duplicates...")
        original_size = len(dataset)
        
        # Create a set to track seen texts
        seen_texts = set()
        
        def is_unique(example):
            text = example.get("text", "")
            if text in seen_texts:
                return False
            seen_texts.add(text)
            return True
        
        unique_dataset = dataset.filter(is_unique, desc="Removing duplicates")
        
        filtered_size = len(unique_dataset)
        logger.info(f"Removed {original_size - filtered_size} duplicates. "
                   f"Remaining: {filtered_size}")
        
        return unique_dataset


def preprocess_medical_dataset(
    dataset_dict: DatasetDict,
    instruction_template: Optional[str] = None,
    max_length: int = 512,
    remove_duplicates: bool = True
) -> DatasetDict:
    """
    Convenience function to preprocess medical dataset.
    
    Args:
        dataset_dict: DatasetDict with train/val/test splits
        instruction_template: Optional custom template
        max_length: Maximum sequence length
        remove_duplicates: Whether to remove duplicate examples
        
    Returns:
        Preprocessed DatasetDict
    """
    preprocessor = DataPreprocessor(instruction_template, max_length)
    
    processed_dict = {}
    
    for split_name, split_data in dataset_dict.items():
        logger.info(f"Processing {split_name} split...")
        
        processed = preprocessor.preprocess_dataset(split_data)
        
        if remove_duplicates:
            processed = preprocessor.remove_duplicates(processed)
        
        processed_dict[split_name] = processed
    
    return DatasetDict(processed_dict)


if __name__ == "__main__":
    # Example usage
    from datasets import Dataset
    
    # Create sample data
    sample_data = {
        "instruction": ["What is diabetes?", "How to treat fever?"],
        "response": [
            "Diabetes is a chronic condition affecting blood sugar regulation.",
            "Fever can be treated with rest, fluids, and medication like acetaminophen."
        ]
    }
    
    dataset = Dataset.from_dict(sample_data)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_dataset(dataset)
    
    print("Original example:")
    print(dataset[0])
    print("\nProcessed example:")
    print(processed[0]["text"])
