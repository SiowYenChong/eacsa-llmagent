import torch
from captum.attr import LayerIntegratedGradients

class EmotionExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Access embeddings through the base RoBERTa model
        self.lig = LayerIntegratedGradients(
            self._forward, 
            self.model.roberta.embeddings.word_embeddings
        )
        
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Custom forward function for Captum compatibility"""
        return self.model(input_ids=input_ids).logits

    def explain(self, text: str) -> dict:
        # Tokenize and prepare inputs
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs.input_ids
        
        # Compute attributions
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            target=1,  # Verify this matches your emotion class index
            return_convergence_delta=True
        )
        
        return {
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids[0]),
            'attributions': attributions.sum(dim=-1).squeeze(0).tolist(),
            'convergence_delta': delta.item()
        }