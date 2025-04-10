import os
import logging
import numpy as np
from typing import Dict, Any, List
from transformers import pipeline

SENTIMENT_MODEL_PATH = "models/sentiment_model/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_PATH = "models/sentiment_model/emotion-english-distilroberta-base"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAgent:
    """Advanced sentiment analysis with multi-model integration and tone guidance"""
    
    def __init__(self):
        self.sentiment_history = []
        self.intensity_window = []
        try:
            # Initialize sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL_PATH
            )
            
            # Verify sentiment model with case-insensitive check
            test_output = self.sentiment_analyzer("I love this!")
            if not isinstance(test_output, list) or len(test_output) == 0:
                raise ValueError("Invalid sentiment model output")
                
            test_result = test_output[0]
            received_label = test_result['label'].lower()
            allowed_labels = {'negative', 'neutral', 'positive'}
            
            if received_label not in allowed_labels:
                raise ValueError(f"Unexpected sentiment label: {test_result['label']}")

            logger.info("Sentiment model initialized successfully")
            
            # Initialize emotion classifier
            self.emotion_classifier = pipeline(
                "text-classification",
                model=EMOTION_MODEL_PATH,
                top_k=None,
                return_all_scores=True
            )
            
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def analyze(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis with emotion tracking"""
        if not text.strip():
            return self._default_analysis()
            
        try:
            sentiment_result = self.sentiment_analyzer(text)[0]
            emotion_results = self.emotion_classifier(text)[0]
            
            analysis = {
                "sentiment": self._parse_sentiment(sentiment_result),
                "emotions": self._parse_emotions(emotion_results),
                "context_shift": self._detect_sentiment_shift(sentiment_result['score']),
                "intensity_trend": self._calculate_intensity_trend(sentiment_result['score']),
                "valence": self._calculate_valence(sentiment_result['score'], emotion_results)
            }
            self.sentiment_history.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return self._default_analysis()

    def _parse_sentiment(self, result: Dict) -> Dict[str, Any]:
        """Convert labels to lowercase and standardize format"""
        return {
            "label": result['label'].lower(),
            "score": float(result['score'])
        }

    def _parse_emotions(self, emotions: List[Dict]) -> List[Dict[str, Any]]:
        """Process emotions with thresholding"""
        MIN_SCORE = 0.3
        return sorted([
            {"label": e['label'], "score": float(e['score'])} 
            for e in emotions if e['score'] >= MIN_SCORE
        ], key=lambda x: x['score'], reverse=True)[:3]

    def _detect_sentiment_shift(self, current_score: float) -> bool:
        """Detect significant sentiment changes"""
        if len(self.sentiment_history) < 2:
            return False
        prev_scores = [s['sentiment']['score'] for s in self.sentiment_history[-2:]]
        return abs(current_score - np.mean(prev_scores)) > 0.4

    def _calculate_intensity_trend(self, current_score: float) -> str:
        """Track intensity changes over recent messages"""
        self.intensity_window.append(current_score)
        self.intensity_window = self.intensity_window[-3:]
        if len(self.intensity_window) < 2:
            return "stable"
        avg_diff = np.mean(np.diff(self.intensity_window))
        return "increasing" if avg_diff > 0.1 else "decreasing" if avg_diff < -0.1 else "stable"

    def _calculate_valence(self, sentiment_score: float, emotions: List[Dict]) -> float:
        """Calculate emotional valence (-1 to 1)"""
        emotion_weights = {
            'anger': -0.9, 'disgust': -0.8, 'fear': -0.7,
            'joy': 0.9, 'neutral': 0.0, 'sadness': -0.8,
            'surprise': 0.4
        }
        primary_emotion = emotions[0]['label'].lower() if emotions else 'neutral'
        return emotion_weights.get(primary_emotion, 0.0) * sentiment_score

    def generate_tone_guidance(self, analysis: Dict) -> Dict[str, Any]:
        """Generate detailed response recommendations"""
        emotion_map = {
            'anger': {'structure': 'apology-first', 'empathy': 4},
            'disgust': {'structure': 'urgent-resolution', 'empathy': 5},
            'joy': {'structure': 'positive-reinforcement', 'empathy': 2},
            'neutral': {'structure': 'general-support', 'empathy': 2}
        }
        primary_emotion = analysis['emotions'][0]['label'].lower() if analysis['emotions'] else 'neutral'
        strategy = emotion_map.get(primary_emotion, emotion_map['neutral'])
        
        return {
            "base_tone": "professional",
            "emotional_strategy": strategy,
            "urgency_level": "high" if analysis['intensity_trend'] == "increasing" else "normal"
        }

    def generate_tone_instruction(self, analysis: Dict[str, Any]) -> str:
        """Generate natural language tone instructions (maintained from original)"""
        guidance = self.generate_tone_guidance(analysis)
        strategy = guidance['emotional_strategy']
        
        instruction = f"Respond using {strategy['structure']} structure. "
        instruction += f"Tone: {guidance['base_tone']}. "
        instruction += f"Empathy level: {strategy['empathy']}/5. "
        instruction += f"Urgency: {guidance['urgency_level'].capitalize()}"
        
        return instruction

    def _default_analysis(self) -> Dict[str, Any]:
        """Fallback for error conditions"""
        return {
            "sentiment": {"label": "neutral", "score": 0.0},
            "emotions": [{"label": "neutral", "score": 0.0}],
            "context_shift": False,
            "intensity_trend": "stable",
            "valence": 0.0
        }

    # Maintained original simple analysis method
    def simple_analyze(self, text: str) -> Dict[str, Any]:
        """Simplified analysis for quick calls"""
        if not text.strip():
            return {"sentiment": "neutral", "emotions": []}
            
        analysis = self.analyze(text)
        return {
            "sentiment": analysis['sentiment']['label'],
            "emotions": [e['label'] for e in analysis['emotions']]
        }