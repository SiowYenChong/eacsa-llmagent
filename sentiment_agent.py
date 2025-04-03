import os
import logging
from typing import Tuple
from transformers import pipeline
# import tensorflow as tf

# # Fix OpenMP conflict for Vercel deployment
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Ensure TensorFlow uses CPU if GPU is not available
# if not tf.config.list_physical_devices('GPU'):
#     tf.config.set_visible_devices([], 'GPU')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAgent:
    """
    Agent responsible for analyzing sentiment and emotion in text using transformer models.
    Uses BERT for sentiment analysis and DistilBERT for emotion classification.
    """

    # Load both models
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    emotion_classifier = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student",
                                  top_k=None)

    def __init__(self):
        """Initialize the sentiment and emotion analysis models"""
        # try:
        #     self.sentiment_analyzer = pipeline(
        #         "sentiment-analysis",
        #         model="nlptown/bert-base-multilingual-uncased-sentiment"
        #     )
        #     self.emotion_classifier = pipeline(
        #         "text-classification",
        #         model="joeddav/distilbert-base-uncased-go-emotions-student",
        #         top_k=None
        #     )
        #     logger.info("Transformers models loaded successfully")
        # except Exception as e:
        #     logger.error(f"Failed to load transformers models: {str(e)}")
        #     raise ImportError("Required transformers models could not be loaded") from e

    def analyze(self, text: str) -> Tuple[str, str]:
        """
        Analyzes sentiment and emotion in text.
        Returns a tuple (sentiment, dominant_emotion).
        """
        if not text or not isinstance(text, str):
            return "neutral", "neutral"

        try:
            return self.analyze_sentiment_and_emotion(text)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return "neutral", "neutral"

    def analyze_sentiment_and_emotion(self, user_input: str) -> Tuple[str, str]:
        """
        Analyzes both sentiment and emotion of user input.

        Returns:
            (sentiment, dominant_emotion)
        """
        try:
            # Sentiment Analysis
            sentiment_results = self.sentiment_analyzer(user_input)
            sentiment_label = sentiment_results[0].get("label", "3 stars")  # Default to neutral

            sentiment_map = {
                "1 star": "negative",
                "2 stars": "negative",
                "3 stars": "neutral",
                "4 stars": "positive",
                "5 stars": "positive"
            }
            sentiment = sentiment_map.get(sentiment_label, "neutral")

            # Emotion Analysis
            emotion_results = self.emotion_classifier(user_input)

            if emotion_results and isinstance(emotion_results, list) and len(emotion_results) > 0:
                sorted_emotions = sorted(emotion_results[0], key=lambda x: x["score"], reverse=True)
                dominant_emotion = sorted_emotions[0].get("label", "neutral")
            else:
                dominant_emotion = "neutral"

            logger.info(f"Analysis result: sentiment={sentiment}, emotion={dominant_emotion}")
            return sentiment, dominant_emotion

        except Exception as e:
            logger.error(f"Error in analyze_sentiment_and_emotion: {str(e)}")
            return "neutral", "neutral"

    def generate_tone_instruction(self, sentiment: str, emotion: str) -> str:
        """
        Generates tone instructions based on detected sentiment and emotion.

        Returns:
            str: Tone instruction for the response agent
        """
        try:
            if sentiment == "negative" or emotion in ["disappointment", "sadness", "anger", "disapproval"]:
                return "Please be patient. I'll do my best to help you with your issue. I'm sorry you're feeling this way, and I'll try to assist in resolving things."
            elif sentiment == "positive" or emotion in ["joy", "excitement", "optimism"]:
                return "I'm so glad to hear you're happy! How can I assist you today with a smile?"
            else:
                return "Let me know how I can assist you. I'm here to help."
        except Exception as e:
            logger.error(f"Error generating tone instruction: {str(e)}")
            return "Let me know how I can assist you. I'm here to help."

