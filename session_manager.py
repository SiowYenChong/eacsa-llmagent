from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self.sessions = []
        logger.info("SessionManager initialized with empty session list")

    def _generate_session_id(self) -> str:
        """Generate UUID4-based session ID"""
        return str(uuid.uuid4())

    def create_session(self, title: str) -> Dict[str, Any]:
        """Create new session with complete metadata"""
        new_session = {
            'id': self._generate_session_id(),
            'title': title,
            'history': [],
            'emotion_timeline': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'message_count': 0
        }
        self.sessions.append(new_session)
        logger.info(f"Created new session: {new_session['id']}")
        return new_session

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session by ID with error handling"""
        if not session_id:
            logger.error("Attempted to access session with empty ID")
            raise ValueError("Session ID cannot be empty")
            
        for session in self.sessions:
            if session['id'] == session_id:
                return session
                
        logger.warning(f"Session {session_id} not found, creating new")
        return self.create_session("Recovered Session")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Get simplified session list for UI display"""
        return [{
            'id': s['id'],
            'title': s['title'],
            'created': s['created_at'],
            'updated': s['last_updated'],
            'message_count': len(s['history'])
        } for s in self.sessions]
            
        return session

    def add_message_to_session(self, session_id: str, role: str, content: str, 
                             sentiment_score: float = None, emotions: list = None):
        """Add message to session with validation"""
        session = self.get_session(session_id)
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'sentiment': {
                'score': sentiment_score,
                'emotions': emotions or []
            } if role == 'user' else None,
            'resolved': False,
            'rating': None
        }
        
        if role == 'assistant':
            message['response_quality'] = {
                'empathy_score': None,
                'resolution_score': None,
                'sentiment_alignment': None
            }
        
        session['history'].append(message)
        session['message_count'] = len(session['history'])
        session['last_updated'] = datetime.now().isoformat()
        logger.info(f"Added {role} message to session {session_id}")

        # Update emotion timeline
        if sentiment_score is not None and emotions:
            timeline_entry = {
                'timestamp': datetime.now().isoformat(),
                'score': sentiment_score,
                'dominant_emotion': emotions[0]['label'] if emotions else 'neutral',
                'emotion_intensity': emotions[0]['score'] if emotions else 0.0
            }
            session['emotion_timeline'].append(timeline_entry)
            logger.debug(f"Updated emotion timeline for {session_id}")

    def mark_message_resolved(self, session_id: str, message_index: int, rating: int = None):
        """Mark message as resolved with optional rating"""
        session = self.get_session(session_id)
        
        if 0 <= message_index < len(session['history']):
            session['history'][message_index]['resolved'] = True
            if rating is not None and 1 <= rating <= 5:
                session['history'][message_index]['rating'] = rating
            session['last_updated'] = datetime.now().isoformat()
            logger.info(f"Marked message {message_index} resolved in {session_id}")
        else:
            logger.error(f"Invalid message index {message_index} for session {session_id}")

    def clear_session(self, session_id: str) -> None:
        """Reset session history while preserving metadata"""
        session = self.get_session(session_id)
        session['history'] = []
        session['emotion_timeline'] = []
        session['message_count'] = 0
        session['last_updated'] = datetime.now().isoformat()
        logger.info(f"Cleared session {session_id}")

    def list_sessions(self) -> List[Dict]:
        """Get session metadata for UI display"""
        return [{
            'id': s['id'],
            'title': s['title'],
            'created': s['created_at'],
            'updated': s['last_updated'],
            'message_count': s['message_count']
        } for s in self.sessions]

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Legacy title-based lookup (not recommended)"""
        return next((s for s in self.sessions if s['title'] == title), None)

    def session_exists(self, session_id: str) -> bool:
        """Check if session ID is valid"""
        return any(s['id'] == session_id for s in self.sessions)