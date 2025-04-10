from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self.sessions = []
        self.current_session = self._create_new_session()
        
    def _create_new_session(self, title: str = "New Session") -> Dict[str, Any]:
        return {
            'id': datetime.now().strftime("%Y%m%d%H%M%S"),
            'title': title,
            'history': [],
            'emotion_timeline': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    def add_message_to_current_session(self, role: str, content: str, 
                                  sentiment_score: float = None, 
                                  emotions: list = None):
        """Store message with full sentiment context"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "sentiment": {
                "score": sentiment_score,
                "emotions": emotions or []
            } if role == "user" else None,
            "feedback": None
        }
        
        if role == "assistant":
            message["response_quality"] = {
                "empathy_score": None,
                "resolution_score": None,
                "sentiment_alignment": None
            }
        
        self.current_session['history'].append(message)
        self._update_emotion_timeline(sentiment_score, emotions)
        
    def _update_emotion_timeline(self, score: float, emotions: list):
        """Track emotional state evolution"""
        if score is None or not emotions:
            return
        
        timeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "dominant_emotion": emotions[0]['label'] if emotions else 'neutral',
            "emotion_intensity": emotions[0]['score'] if emotions else 0.0
        }
        self.current_session['emotion_timeline'].append(timeline_entry)

    def get_all_session_titles(self) -> List[str]:
        return [s['title'] for s in self.sessions]

    def mark_message_resolved(self, message_index: int, rating: int = None):
        """Unified resolution handling"""
        if 0 <= message_index < len(self.current_session['history']):
            self.current_session['history'][message_index]['resolved'] = True
            if rating:
                self.current_session['history'][message_index]['rating'] = rating
            self.current_session['last_updated'] = datetime.now().isoformat()

    def create_session(self, title: str) -> Dict[str, Any]:
        new_session = self._create_new_session(title)
        self.sessions.append(new_session)
        self.current_session = new_session
        return new_session

    def clear_current_session(self) -> None:
        self.current_session['history'] = []
        self.current_session['last_updated'] = datetime.now().isoformat()

    # Add to SessionManager class in session_manager.py
    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get session by exact title match"""
        return next((s for s in self.sessions if s['title'] == title), None)
    
