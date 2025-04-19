from datetime import datetime, timedelta

class HITLManager:
    ESCALATION_THRESHOLDS = {
        'anger': 0.85,
        'frustration_duration': timedelta(minutes=5),
        'financial_impact': 500.00
    }
    
    def __init__(self, session_manager, current_session_id):
        self.session_manager = session_manager
        self.current_session_id = current_session_id

    def check_escalation_needed(self):
        """Determine if human intervention required"""
        current_session = self.session_manager.get_session(self.current_session_id)
        timeline = current_session['emotion_timeline']  # Fixed variable name
        
        # Check recent strong negative emotions
        recent_anger = any(
            e['dominant_emotion'] == 'anger' and e['score'] > self.ESCALATION_THRESHOLDS['anger']
            for e in timeline[-3:]
        )
        
        # Check prolonged frustration
        time_window = datetime.now() - self.ESCALATION_THRESHOLDS['frustration_duration']
        frustration_count = sum(
            1 for e in timeline 
            if e['dominant_emotion'] == 'frustration' 
            and datetime.fromisoformat(e['timestamp']) > time_window
        )
        
        return recent_anger or frustration_count >= 3