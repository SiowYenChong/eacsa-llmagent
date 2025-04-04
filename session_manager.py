from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Enhanced session manager with automatic training data collection
    """
    
    def __init__(self):
        self.sessions = []
        self.current_session = self._create_new_session()
        self.training_queue = []  # For background training processing
        
    def _create_new_session(self, title: str = "New Session") -> Dict[str, Any]:
        """Create a new session with extended metadata"""
        now = datetime.now()
        return {
            'id': now.strftime("%Y%m%d%H%M%S"),
            'title': title,
            'history': [],
            'created_at': now.isoformat(),
            'last_updated': now.isoformat(),
            'user_rating': None,
            'duration': 0.0,
            'metadata': {
                'interaction_count': 0,
                'knowledge_used': []
            }
        }
        
    def add_rating_to_current_session(self, stars: int) -> None:
        """Store user rating and queue for background training"""
        self.current_session['user_rating'] = {
            'stars': stars,
            'rated_at': datetime.now().isoformat()
        }
        self._add_to_training_queue()
        self._update_session_metadata()
        
    def _add_to_training_queue(self) -> None:
        """Add rated session to background training queue"""
        if self.current_session.get('user_rating'):
            session_id = self.current_session['id']
            if session_id not in self.training_queue:
                self.training_queue.append(session_id)
                logger.info(f"Added session {session_id} to training queue")
        
    def process_training_queue(self) -> None:
        """Process all sessions in the training queue (call periodically)"""
        while self.training_queue:
            session_id = self.training_queue.pop(0)
            self._save_training_data(session_id)
            
    def _save_training_data(self, session_id: str) -> None:
        """Internal method to save training data for a session"""
        try:
            session = self.get_session_by_id(session_id)
            if not session or not session.get('user_rating'):
                return

            training_data = {
                "conversation": session['history'],
                "rating": session['user_rating'],
                "metadata": session['metadata'],
                "timestamps": {
                    "created": session['created_at'],
                    "duration": session['duration']
                }
            }
            
            os.makedirs("training_data", exist_ok=True)
            with open(f"training_data/session_{session_id}.json", "w") as f:
                json.dump(training_data, f, indent=2)
            
            logger.info(f"Saved training data for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving training data for {session_id}: {str(e)}")
        
    def create_session(self, title: str) -> Dict[str, Any]:
        """Create and store a new session with tracking"""
        new_session = self._create_new_session(title)
        self.sessions.append(new_session)
        self.current_session = new_session
        return new_session
        
    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by unique ID"""
        return next((s for s in self.sessions if s['id'] == session_id), None)
        
    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Find session by title"""
        return next((s for s in self.sessions if s['title'] == title), None)
        
    def set_current_session(self, session: Dict[str, Any]) -> None:
        """Set active session"""
        self.current_session = session
        
    def add_message_to_current_session(self, role: str, content: str) -> Dict[str, Any]:
        """
        Add a message to the current session with automatic timestamp.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()  # Auto-generated timestamp
        }
        self.current_session['history'].append(message)
        self._update_session_metadata()
        return message
        
    def add_rating_to_current_session(self, stars: int) -> None:
        """Store user rating for current session"""
        self.current_session['user_rating'] = {
            'stars': stars,
            'rated_at': datetime.now().isoformat(),
            'calculated_sentiment': self.current_session.get('metadata', {}).get('sentiment')
        }
        self._update_session_metadata()
        
    def save_training_data(self, session_id: str) -> bool:
        """Export session data only if rated"""
        session = self.get_session_by_id(session_id)
        if not session or not session.get('user_rating'):
            return False  # Skip unrated sessions
            
        try:
            training_data = {
                "conversation": session['history'],
                "rating": session['user_rating'],
                "metadata": session['metadata'],
                "timestamps": {
                    "created": session['created_at'],
                    "duration": session['duration']
                }
            }
            
            os.makedirs("training_data", exist_ok=True)
            with open(f"training_data/session_{session_id}.json", "w") as f:
                json.dump(training_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving training data: {str(e)}")
            return False
        
    def _update_session_metadata(self) -> None:
        """Update session statistics and metadata"""
        now = datetime.now()
        created = datetime.fromisoformat(self.current_session['created_at'])
        self.current_session['duration'] = (now - created).total_seconds()
        self.current_session['last_updated'] = now.isoformat()
        self.current_session['metadata']['interaction_count'] = len(self.current_session['history'])
        
    def clear_current_session(self) -> None:
        """Reset current session while preserving metadata"""
        self.current_session['history'] = []
        self._update_session_metadata()
        
    def get_all_session_titles(self) -> List[str]:
        """Get list of all session titles"""
        return [s['title'] for s in self.sessions]

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get full history for a session"""
        session = self.get_session_by_id(session_id)
        return session['history'] if session else []
    def mark_message_resolved(self, message_index: int, rating: int = None):
        """Mark a message as resolved and optionally rate it"""
        if 0 <= message_index < len(self.current_session['history']):
            self.current_session['history'][message_index]['resolved'] = True
            if rating is not None:
                self.current_session['history'][message_index]['rating'] = rating
            self._update_session_metadata()