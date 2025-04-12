import numpy as np
import json
from sklearn.metrics import accuracy_score

class BiasAuditor:
    def __init__(self, reference_data_path: str = 'data/equity_baseline.json'):
        with open(reference_data_path) as f:
            self.baseline = json.load(f)
        
    def audit(self, predictions: list, ground_truth: list) -> dict:
        disparity = {}
        for group in self.baseline['demographics']:
            group_mask = [g['group'] == group for g in ground_truth]
            group_acc = accuracy_score(
                [p['label'] for p in predictions[group_mask]],
                [g['label'] for g in ground_truth[group_mask]]
            )
            disparity[group] = abs(group_acc - self.baseline['threshold'])
        return disparity