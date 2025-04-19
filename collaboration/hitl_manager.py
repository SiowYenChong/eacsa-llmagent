from datetime import datetime, timedelta
class HITLManager:
    ESCALATION_THRESHOLDS = {
        'anger': 0.85,
        'frustration_duration': timedelta(minutes=5),
        'financial_impact': 500.00
    }
    
    def __init__(self, session_manager):  # Remove current_session_id parameter
        self.session_manager = session_manager

    def check_escalation_needed(self, session_id):  # Add session_id parameter
        """Determine if human intervention required"""
        current_session = self.session_manager.get_session(session_id)
        timeline = current_session['emotion_timeline']  
        
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