import os
import json
from datetime import datetime

def save_training_data(session_data: dict):
    """Save complete session data for training"""
    try:
        training_data = {
            "conversation": session_data.get("messages", []),
            "user_rating": session_data.get("rating", {}),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "session_id": session_data.get("id"),
                "duration": session_data.get("duration", 0)
            }
        }
        
        os.makedirs("training_data", exist_ok=True)
        filename = f"training_data/session_{session_data['id']}.json"
        
        with open(filename, "w") as f:
            json.dumps(training_data, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving training data: {str(e)}")
        return False