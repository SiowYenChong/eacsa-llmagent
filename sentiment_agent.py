import os
import logging
from typing import Dict, Any, List
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAgent:
    """Handles sentiment analysis and emotion detection for response tone adaptation"""
    
    def __init__(self):
        try:
            # Initialize both models with specified pretrained models
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            self.emotion_classifier = pipeline(
                "text-classification",
                model="joeddav/distilbert-base-uncased-go-emotions-student",
                top_k=None
            )
            logger.info("Both sentiment and emotion models loaded successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze both sentiment and emotions in text"""
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment": {"label": "neutral", "score": 0.0},
                "emotions": [{"label": "neutral", "score": 0.0}]
            }
            
        try:
            return {
                "sentiment": self._analyze_sentiment(text),
                "emotions": self._analyze_emotions(text)
            }
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "sentiment": {"label": "neutral", "score": 0.0},
                "emotions": [{"label": "neutral", "score": 0.0}]
            }

    def _analyze_emotions(self, text: str) -> List[Dict[str, Any]]:
        """Detailed emotion analysis returning top 3 emotions"""
        try:
            results = self.emotion_classifier(text)[0]
            return sorted([
                {
                    "label": e.get("label", "neutral"),
                    "score": float(e.get("score", 0.0))
                } for e in results
            ], key=lambda x: x['score'], reverse=True)[:3]
        except Exception as e:
            logger.error(f"Emotion analysis error: {str(e)}")
            return [{"label": "neutral", "score": 0.0}]

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Sentiment analysis with label normalization"""
        try:
            result = self.sentiment_analyzer(text)[0]
            # Convert star rating to sentiment labels
            stars = int(result['label'].split()[0])
            if stars <= 2:
                label = "negative"
            elif stars == 3:
                label = "neutral"
            else:
                label = "positive"
            
            return {
                "label": label,
                "score": float(result['score'])
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {"label": "neutral", "score": 0.0}

    def generate_tone_instruction(self, analysis: Dict[str, Any]) -> str:
        """Generate tone based on emotions and sentiment"""
        try:
            # Use both sentiment and emotions for tone decision
            if analysis["sentiment"]["label"] == "negative":
                if any(e["label"] in ["anger", "disappointment"] for e in analysis["emotions"][:1]):
                    return "Empathetic and apologetic tone"
                return "Understanding and solution-oriented tone"
            
            if analysis["sentiment"]["label"] == "positive":
                if any(e["label"] in ["excitement", "joy"] for e in analysis["emotions"][:1]):
                    return "Enthusiastic and congratulatory tone"
                return "Positive and encouraging tone"
            
            return "Professional and neutral tone"
        except Exception as e:
            logger.error(f"Tone generation error: {str(e)}")
            return "Professional and neutral tone"