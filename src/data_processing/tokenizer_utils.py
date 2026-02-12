"""
Tokenizer utilities for preparing data for model training.
"""

from transformers import AutoTokenizer
from typing import Dict, List, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizerHelper:
    """
    Helper class for tokenization operations.
    """
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        Initialize tokenizer helper.
        
        Args:
            model_name: Name of the model to load tokenizer from
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize examples for training.
        
        Args:
            examples: Dictionary with 'text' key containing formatted examples
            
        Returns:
            Dictionary with tokenized inputs
        """
        tokenized = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def tokenize_dataset(self, dataset, num_proc: Optional[int] = 4):
        """
        Tokenize entire dataset.
        
        Args:
            dataset: Dataset to tokenize
            num_proc: Number of processes for parallel processing
            
        Returns:
            Tokenized dataset
        """
        logger.info("Tokenizing dataset...")
        
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        logger.info(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def get_token_statistics(self, dataset) -> Dict:
        """
        Get statistics about token lengths in dataset.
        
        Args:
            dataset: Dataset with 'text' field
            
        Returns:
            Dictionary with statistics
        """
        lengths = []
        
        for example in dataset:
            tokens = self.tokenizer(example["text"], truncation=False)
            lengths.append(len(tokens["input_ids"]))
        
        import numpy as np
        
        stats = {
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "std": np.std(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
            "p95": np.percentile(lengths, 95),
            "p99": np.percentile(lengths, 99)
        }
        
        return stats
    
    def encode_single(self, text: str) -> Dict:
        """
        Encode a single text example.
        
        Args:
            text: Input text
            
        Returns:
            Encoded dictionary
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
    
    def decode_tokens(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def create_tokenizer(model_name: str, max_length: int = 512) -> TokenizerHelper:
    """
    Convenience function to create tokenizer helper.
    
    Args:
        model_name: Name of the model
        max_length: Maximum sequence length
        
    Returns:
        TokenizerHelper instance
    """
    return TokenizerHelper(model_name, max_length)


if __name__ == "__main__":
    # Example usage
    tokenizer_helper = create_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Test tokenization
    text = "### Instruction:\nWhat is diabetes?\n\n### Response:\nDiabetes is a chronic condition."
    encoded = tokenizer_helper.encode_single(text)
    
    print(f"Original text: {text}")
    print(f"\nEncoded shape: {encoded['input_ids'].shape}")
    print(f"Token IDs (first 20): {encoded['input_ids'][0][:20].tolist()}")
    
    # Decode
    decoded = tokenizer_helper.decode_tokens(encoded['input_ids'][0].tolist())
    print(f"\nDecoded text: {decoded}")
