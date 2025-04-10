from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from typing import Dict

class DataSanitizer:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def sanitize_text(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language='en')
        return self.anonymizer.anonymize(text, results).text

    def sanitize_session(self, session: Dict) -> Dict:
        return {
            **session,
            'history': [self._sanitize_message(m) for m in session['history']]
        }

    def _sanitize_message(self, message: Dict) -> Dict:
        return {
            **message,
            'content': self.sanitize_text(message['content'])
        }