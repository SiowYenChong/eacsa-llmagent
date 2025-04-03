import os
import json
import time
from datetime import datetime
import logging

import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Customer Service Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy imports to handle missing dependencies gracefully
def import_agents():
    """Import our agent modules with error handling"""
    try:
        from sentiment_agent import SentimentAgent
        from knowledge_agent import KnowledgeAgent
        from llm_agent import LLMAgent
        from session_manager import SessionManager
        return SentimentAgent, KnowledgeAgent, LLMAgent, SessionManager
    except ImportError as e:
        st.error(f"Failed to import required modules: {str(e)}")
        st.stop()

# Import our agent modules
SentimentAgent, KnowledgeAgent, LLMAgent, SessionManager = import_agents()

# Initialize session state for our agents and manager
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
if 'knowledge_agent' not in st.session_state:
    st.session_state.knowledge_agent = None
if 'sentiment_agent' not in st.session_state:
    st.session_state.sentiment_agent = None
if 'llm_agent' not in st.session_state:
    st.session_state.llm_agent = None

# Initialize our agents
@st.cache_resource
def initialize_agents():
    """Initialize all agents needed for the application"""
    try:
        knowledge_agent = KnowledgeAgent()
        sentiment_agent = SentimentAgent()
        llm_agent = LLMAgent(model_name="gpt-4-turbo-preview")
        return knowledge_agent, sentiment_agent, llm_agent
    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}")
        st.error(f"Error initializing agents: {str(e)}")
        # Return None objects so the app can still run with limited functionality
        return None, None, None

# Initialize agents if needed
if st.session_state.knowledge_agent is None:
    with st.spinner("Initializing agents..."):
        try:
            knowledge_agent, sentiment_agent, llm_agent = initialize_agents()
            st.session_state.knowledge_agent = knowledge_agent
            st.session_state.sentiment_agent = sentiment_agent
            st.session_state.llm_agent = llm_agent
        except Exception as e:
            st.error(f"Failed to initialize agents: {str(e)}")
            st.warning("Some features may be limited. Please check your dependencies.")

# Function to process user input and generate response
def process_user_input(user_query):
    """Process user input through our agent system"""
    # Add user message to current session history
    user_message = st.session_state.session_manager.add_message_to_current_session(
        role="user",
        content=user_query
    )
    
    # Get relevant context using knowledge agent
    context = "No knowledge base available."
    if st.session_state.knowledge_agent:
        with st.spinner("🔍 Looking up information..."):
            try:
                context = st.session_state.knowledge_agent.get_context(user_query)
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                context = f"Error retrieving knowledge: {str(e)}"
    
    # Process sentiment and emotion using sentiment agent
    tone_instruction = ""
    if st.session_state.sentiment_agent:
        with st.spinner("💭 Analyzing sentiment..."):
            try:
                sentiment, emotion = st.session_state.sentiment_agent.analyze(user_query)
                tone_instruction = st.session_state.sentiment_agent.generate_tone_instruction(sentiment, emotion)
                print(sentiment, '\n', emotion)
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {str(e)}")
    
    # Generate response with LLM agent
    assistant_message_content = "I'm sorry, but I'm having trouble generating a response right now."
    if st.session_state.llm_agent:
        with st.spinner("💭 Generating response..."):
            try:
                assistant_message_content = st.session_state.llm_agent.generate_response(
                    query=user_query,
                    context=context,
                    history=st.session_state.session_manager.current_session['history'],
                    tone_instruction=tone_instruction
                )
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                assistant_message_content = f"I'm sorry, but I encountered an error: {str(e)}"
        
    # Add assistant message to history
    assistant_message = st.session_state.session_manager.add_message_to_current_session(
        role="assistant",
        content=assistant_message_content
    )
    
    return assistant_message

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center;'>
        <h2>Customer Service Bot</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.write("### Bot Configuration")
    st.info("""
    **AI Customer Service**
    Version: 1.0.0
    Model: GPT-4 Turbo
    RAG-enabled for accurate responses
    """)

    st.markdown("---")

    # Session management
    session_titles = st.session_state.session_manager.get_all_session_titles()
    session_titles.append("🆕 New Session")

    selected_session_title = st.selectbox("Select a Session", session_titles)

    if selected_session_title == "🆕 New Session":
        new_session_title = st.text_input("Name your session:", value=f"Session {len(session_titles)}")
        if st.button("Create Session"):
            st.session_state.session_manager.create_session(new_session_title)
            st.rerun()
    else:
        session = st.session_state.session_manager.get_session_by_title(selected_session_title)
        if session:
            st.session_state.session_manager.set_current_session(session)

    if st.button("🧹 Clear Current Session"):
        st.session_state.session_manager.clear_current_session()
        st.rerun()

# Main interface
st.markdown("""
<h1 style='text-align: center;'>
    🤖 Customer Service Assistant
</h1>
<p style='text-align: center;'>Ask me anything about our products and services</p>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.session_manager.current_session['history']:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])
        if role == "assistant":
            st.caption(f"*Bot • {message['timestamp']}*")

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process input through our agent system
    assistant_message = process_user_input(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_message["content"])
        st.caption(f"Generated at: {assistant_message['timestamp']}")

# Create requirements.txt for Vercel deployment
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("""
streamlit==1.30.0
openai==1.6.1
langchain==0.1.0
langchain-community==0.0.13
faiss-cpu==1.7.4
python-dotenv==1.0.0
transformers>=4.30.0
torch>=1.13.0
sympy==1.12.0
""")

# Create vercel.json for deployment configuration
if not os.path.exists("vercel.json"):
    with open("vercel.json", "w") as f:
        json.dump({
            "builds": [
                {
                    "src": "app.py",
                    "use": "@vercel/python"
                }
            ],
            "routes": [
                {
                    "src": "/(.*)",
                    "dest": "app.py"
                }
            ]
        }, f, indent=2)
