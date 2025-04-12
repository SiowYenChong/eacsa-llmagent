from transformers import pipeline

class CodeSwitchHandler:
    def __init__(self):
        self.models = {
            'en-es': pipeline('text-classification', model='CodeSwitchEmo/en-es'),
            'zh-en': pipeline('text-classification', model='CodeSwitchEmo/zh-en')
        }
    
    def analyze_code_switched(self, text: str, lang_pair: str) -> dict:
        if lang_pair not in self.models:
            raise ValueError(f"Unsupported language pair: {lang_pair}")
        return self.models[lang_pair](text)