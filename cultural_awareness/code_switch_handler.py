from transformers import pipeline
import re
import logging
import language_detector 
from typing import Dict, Any
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeSwitchHandler:
    def __init__(self):
        self.patterns = {
            'en-es': re.compile(
                r'\b\w+?(?:ar|er|ir|ado|iendo)\b', 
                re.IGNORECASE
            )
        }
        self.language_detector = language_detector.LanguageDetector()
    
    def detect_code_switch(self, text: str) -> Dict[str, Any]:
        """Detects code-switching using regex patterns and language detection."""
        try:
            # Check for English-Spanish code-switching patterns
            if self.patterns['en-es'].search(text):
                return {'lang_pair': 'en-es', 'confidence': 0.80}
            
            # Detect language and potential code-switching
            lang_info = self.language_detector.detect_language(text)
            
            if lang_info['is_code_switched']:
                primary = lang_info['primary_lang']
                # Determine secondary language based on script detection
                secondary = 'zh' if primary == 'en' else 'en'
                return {
                    'lang_pair': f"{primary}-{secondary}",
                    'confidence': lang_info['confidence']
                }
            
            # No code-switching detected
            return {
                'lang_pair': lang_info['primary_lang'],
                'confidence': lang_info['confidence']
            }
            
        except Exception as e:
            logger.error(f"Code switch detection error: {str(e)}")
            return {'lang_pair': 'en', 'confidence': 0.0}

    def handle_code_switch(self, text: str) -> Dict[str, Any]:
        """Processes text and returns code-switching handling results."""
        detection = self.detect_code_switch(text)
        lang_pair = detection['lang_pair']
        if '-' in lang_pair:
            primary, secondary = lang_pair.split('-')
        else:
            primary = lang_pair
            secondary = None
        
        return {
            'strategy': 'hybrid',
            'primary_lang': primary,
            'secondary_lang': secondary,
            'confidence': detection['confidence']
        }