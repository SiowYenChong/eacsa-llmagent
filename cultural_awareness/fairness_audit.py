import numpy as np
import json
import logging
from sklearn.metrics import accuracy_score
from pathlib import Path

logger = logging.getLogger(__name__)

class BiasAuditor:
    DEFAULT_BASELINE = {
        "demographics": ["general"],
        "threshold": 0.80,
        "reference_accuracy": {"general": 0.80}
    }

    def __init__(self, reference_data_path: str = 'data/equity_baseline.json'):
        self.baseline = self.DEFAULT_BASELINE
        try:
            path = Path(reference_data_path)
            if path.exists():
                with open(path) as f:
                    self.baseline = json.load(f)
            else:
                logger.warning(f"Baseline file not found at {path}, using default values")
        except Exception as e:
            logger.error(f"Error loading baseline: {str(e)}")
            logger.info("Using default bias audit parameters")

    def audit(self, predictions: list, ground_truth: list) -> dict:
        disparity = {}
        try:
            for group in self.baseline['demographics']:
                group_mask = [g.get('group', 'general') == group for g in ground_truth]
                if not any(group_mask):  # Handle missing groups
                    disparity[group] = 0.0
                    continue
                    
                group_acc = accuracy_score(
                    [p.get('label', 0) for p in predictions],
                    [g.get('label', 0) for g in ground_truth]
                )
                disparity[group] = abs(group_acc - self.baseline.get('reference_accuracy', {}).get(group, 0.8))
        except Exception as e:
            logger.error(f"Audit failed: {str(e)}")
            return {"error": str(e)}
        
        return disparity