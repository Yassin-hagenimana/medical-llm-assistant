"""
Evaluation metrics for LLM fine-tuning.
"""

import evaluate
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates various NLP metrics for model evaluation.
    """
    
    def __init__(self):
        """Initialize metrics."""
        logger.info("Loading evaluation metrics...")
        
        try:
            self.bleu = evaluate.load("bleu")
            logger.info("BLEU metric loaded")
        except Exception as e:
            logger.warning(f"Could not load BLEU: {e}")
            self.bleu = None
        
        try:
            self.rouge = evaluate.load("rouge")
            logger.info("ROUGE metric loaded")
        except Exception as e:
            logger.warning(f"Could not load ROUGE: {e}")
            self.rouge = None
        
        try:
            self.meteor = evaluate.load("meteor")
            logger.info("METEOR metric loaded")
        except Exception as e:
            logger.warning(f"Could not load METEOR: {e}")
            self.meteor = None
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with BLEU scores
        """
        if self.bleu is None:
            return {"bleu": 0.0}
        
        # Format references as list of lists
        formatted_refs = [[ref] for ref in references]
        
        try:
            results = self.bleu.compute(
                predictions=predictions,
                references=formatted_refs
            )
            return {
                "bleu": results.get("bleu", 0.0),
                "bleu_1": results.get("precisions", [0])[0] if results.get("precisions") else 0.0,
                "bleu_2": results.get("precisions", [0, 0])[1] if len(results.get("precisions", [])) > 1 else 0.0
            }
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return {"bleu": 0.0}
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        if self.rouge is None:
            return {}
        
        try:
            results = self.rouge.compute(
                predictions=predictions,
                references=references
            )
            return {
                "rouge1": results.get("rouge1", 0.0),
                "rouge2": results.get("rouge2", 0.0),
                "rougeL": results.get("rougeL", 0.0),
                "rougeLsum": results.get("rougeLsum", 0.0)
            }
        except Exception as e:
            logger.error(f"Error computing ROUGE: {e}")
            return {}
    
    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute METEOR score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with METEOR score
        """
        if self.meteor is None:
            return {}
        
        try:
            results = self.meteor.compute(
                predictions=predictions,
                references=references
            )
            return {"meteor": results.get("meteor", 0.0)}
        except Exception as e:
            logger.error(f"Error computing METEOR: {e}")
            return {}
    
    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from loss.
        
        Args:
            loss: Average loss value
            
        Returns:
            Perplexity value
        """
        return np.exp(loss)
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        loss: float = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            loss: Optional loss value for perplexity
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Computing all metrics...")
        
        all_metrics = {}
        
        # BLEU
        bleu_scores = self.compute_bleu(predictions, references)
        all_metrics.update(bleu_scores)
        
        # ROUGE
        rouge_scores = self.compute_rouge(predictions, references)
        all_metrics.update(rouge_scores)
        
        # METEOR
        meteor_scores = self.compute_meteor(predictions, references)
        all_metrics.update(meteor_scores)
        
        # Perplexity
        if loss is not None:
            all_metrics["perplexity"] = self.compute_perplexity(loss)
        
        logger.info("Metrics computed successfully")
        return all_metrics


def calculate_metrics(
    predictions: List[str],
    references: List[str],
    loss: float = None
) -> Dict[str, float]:
    """
    Convenience function to calculate all metrics.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        loss: Optional loss value
        
    Returns:
        Dictionary with all metrics
    """
    calculator = MetricsCalculator()
    return calculator.compute_all_metrics(predictions, references, loss)


if __name__ == "__main__":
    # Example usage
    predictions = [
        "Diabetes is a chronic condition affecting blood sugar.",
        "Aspirin can cause stomach upset."
    ]
    
    references = [
        "Diabetes is a chronic disease that affects blood glucose levels.",
        "Common side effects of aspirin include stomach irritation."
    ]
    
    metrics = calculate_metrics(predictions, references, loss=0.5)
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
