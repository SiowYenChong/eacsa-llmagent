from langdetect import detect, LangDetectException
from typing import Dict, Any
import regex as re

class LanguageDetector:
    def __init__(self):
        self.code_switch_pattern = re.compile(
            r'\p{Script=Latin}+\p{Script=Han}+', 
            re.UNICODE
        )

    def detect_language(self, text: str) -> Dict[str, Any]:
        try:
            lang_code = detect(text)
            is_code_switched = bool(self.code_switch_pattern.search(text))
            return {
                'primary_lang': lang_code,
                'is_code_switched': is_code_switched,
                'confidence': 0.95  # Placeholder confidence value
            }
        except LangDetectException:
            return {'primary_lang': 'en', 'is_code_switched': False, 'confidence': 0.0}

