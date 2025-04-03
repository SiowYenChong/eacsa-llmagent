from datetime import datetime
from typing import List, Dict, Any, Optional

class SessionManager:
    """
    Manages conversation sessions including creating, storing, and retrieving sessions.
    """
    
    def __init__(self):
        """Initialize the session manager with an empty sessions list"""
        self.sessions = []
        self.current_session = self._create_new_session()
        
    def _create_new_session(self, title: str = "New Session") -> Dict[str, Any]:
        """
        Create a new conversation session.
        
        Args:
            title (str): Title for the new session
            
        Returns:
            Dict: New session object
        """
        return {
            'id': datetime.now().strftime("%Y%m%d%H%M%S"),
            'title': title,
            'history': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def create_session(self, title: str) -> Dict[str, Any]:
        """
        Create and store a new session.
        
        Args:
            title (str): Title for the new session
            
        Returns:
            Dict: The newly created session
        """
        new_session = self._create_new_session(title)
        self.sessions.append(new_session)
        self.current_session = new_session
        return new_session
        
    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Find and return a session by its title.
        
        Args:
            title (str): Title of the session to find
            
        Returns:
            Dict or None: The session if found, None otherwise
        """
        for session in self.sessions:
            if session['title'] == title:
                return session
        return None
        
    def set_current_session(self, session: Dict[str, Any]) -> None:
        """
        Set the current active session.
        
        Args:
            session (Dict): The session to set as current
        """
        self.current_session = session
        
    def add_message_to_current_session(self, role: str, content: str) -> Dict[str, Any]:
        """
        Add a message to the current session.
        
        Args:
            role (str): The role of the message sender (user/assistant)
            content (str): The message content
            
        Returns:
            Dict: The created message object
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.current_session['history'].append(message)
        return message
        
    def clear_current_session(self) -> None:
        """Clear the history of the current session."""
        self.current_session['history'] = []
        
    def get_all_session_titles(self) -> List[str]:
        """
        Get titles of all sessions.
        
        Returns:
            List[str]: List of all session titles
        """
        return [session['title'] for session in self.sessions]
