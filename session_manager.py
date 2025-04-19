from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self.sessions = []
        self.current_session_id = None

    def _generate_session_id(self) -> str:
        """Generate unique session ID using UUID"""
        return str(uuid.uuid4())

    def create_session(self, title: str) -> Dict[str, Any]:
        """Create and return new session with UUID"""
        new_session = {
            'id': self._generate_session_id(),
            'title': title,
            'history': [],
            'emotion_timeline': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        self.sessions.append(new_session)
        self.current_session_id = new_session['id']
        return new_session

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session by ID with fallback creation"""
        if not session_id:
            return self.create_session("New Session")
            
        return next(
            (s for s in self.sessions if s['id'] == session_id),
            self.create_session("New Session")
        )

    def add_message_to_session(self, session_id: str, role: str, content: str, 
                             sentiment_score: float = None, emotions: list = None):
        """Add message to specific session with emotion tracking"""
        session = self.get_session(session_id)
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
        
        session['history'].append(message)
        session['last_updated'] = datetime.now().isoformat()
        
        # Update emotion timeline if available
        if sentiment_score is not None and emotions:
            timeline_entry = {
                "timestamp": datetime.now().isoformat(),
                "score": sentiment_score,
                "dominant_emotion": emotions[0]['label'] if emotions else 'neutral',
                "emotion_intensity": emotions[0]['score'] if emotions else 0.0
            }
            session['emotion_timeline'].append(timeline_entry)

    def mark_message_resolved(self, session_id: str, message_index: int, rating: int = None):
        """Mark message as resolved in specific session"""
        session = self.get_session(session_id)
        if 0 <= message_index < len(session['history']):
            session['history'][message_index]['resolved'] = True
            if rating:
                session['history'][message_index]['rating'] = rating
            session['last_updated'] = datetime.now().isoformat()

    def clear_session(self, session_id: str) -> None:
        """Clear specific session's history"""
        session = self.get_session(session_id)
        session['history'] = []
        session['emotion_timeline'] = []
        session['last_updated'] = datetime.now().isoformat()

    def list_sessions(self) -> List[Dict]:
        """Get simplified session list for UI display"""
        return [{
            'id': s['id'],
            'title': s['title'],
            'created': s['created_at'],
            'updated': s['last_updated'],
            'message_count': len(s['history'])
        } for s in self.sessions]

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Legacy method - prefer ID-based access"""
        return next((s for s in self.sessions if s['title'] == title), None)