import torch
from captum.attr import LayerIntegratedGradients

class EmotionExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.lig = LayerIntegratedGradients(model, model.embeddings)
        
    def explain(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors='pt')
        attributions = self.lig.attribute(
            inputs.input_ids,
            target=1  # Emotion class index
        )
        return {
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
            'attributions': attributions.tolist()
        }