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
    """Import agent modules with error handling"""
    try:
        from sentiment_agent import SentimentAgent
        from knowledge_agent import KnowledgeAgent
        from llm_agent import LLMAgent
        from session_manager import SessionManager
        return SentimentAgent, KnowledgeAgent, LLMAgent, SessionManager
    except ImportError as e:
        st.error(f"Failed to import required modules: {str(e)}")
        st.stop()

# Import agent modules
SentimentAgent, KnowledgeAgent, LLMAgent, SessionManager = import_agents()

# Initialize session state
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
if 'knowledge_agent' not in st.session_state:
    st.session_state.knowledge_agent = None
if 'sentiment_agent' not in st.session_state:
    st.session_state.sentiment_agent = None
if 'llm_agent' not in st.session_state:
    st.session_state.llm_agent = None

# Initialize agents
@st.cache_resource
def initialize_agents():
    """Initialize application agents"""
    try:
        return (
            KnowledgeAgent(),
            SentimentAgent(),
            LLMAgent(model_name="gpt-4-turbo-preview"),
            SessionManager()
        )
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        st.error("System initialization failed. Please check logs.")
        st.stop()

if st.session_state.knowledge_agent is None:
    with st.spinner("Loading AI components..."):
        try:
            knowledge_agent, sentiment_agent, llm_agent, session_manager = initialize_agents()
            st.session_state.knowledge_agent = knowledge_agent
            st.session_state.sentiment_agent = sentiment_agent
            st.session_state.llm_agent = llm_agent
            st.session_state.session_manager = session_manager
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()

def display_rating_buttons(message_index: int):
    """Display rating buttons for a specific resolved message"""
    st.markdown("---")
    st.subheader("Rate this resolution")
    
    cols = st.columns(5)
    with cols[0]: st.button("⭐", key=f"{message_index}_1", on_click=lambda: set_rating(message_index, 1))
    with cols[1]: st.button("⭐⭐", key=f"{message_index}_2", on_click=lambda: set_rating(message_index, 2))
    with cols[2]: st.button("⭐⭐⭐", key=f"{message_index}_3", on_click=lambda: set_rating(message_index, 3))
    with cols[3]: st.button("⭐⭐⭐⭐", key=f"{message_index}_4", on_click=lambda: set_rating(message_index, 4))
    with cols[4]: st.button("⭐⭐⭐⭐⭐", key=f"{message_index}_5", on_click=lambda: set_rating(message_index, 5))

def set_rating(message_index: int, rating: int):
    """Store rating for specific resolved message"""
    try:
        st.session_state.session_manager.mark_message_resolved(message_index, rating)
        st.success("Thank you for your feedback! 💌")
        # Remove experimental_rerun()
    except Exception as e:
        logger.error(f"Error saving rating: {str(e)}")
        st.error("Failed to save your rating. Please try again.")

def process_user_input(user_query: str):
    """Process user query through AI pipeline"""
    # Add user message to history
    st.session_state.session_manager.add_message_to_current_session(
        role="user",
        content=user_query
    )
    
    # Retrieve knowledge context
    context = {"text": "", "sources": []}
    if st.session_state.knowledge_agent:
        with st.status("🔍 Looking up information...", expanded=True) as status:
            try:
                context = st.session_state.knowledge_agent.get_context(user_query)
                st.markdown("### Retrieved Information")
                if context.get("sources"):
                    for idx, source in enumerate(context["sources"][:3]):
                        st.markdown(f"""
                        **Match {idx+1}**  
                        Score: `{source['score']:.2f}`  
                        Content:  
                        ```{source['content'][:200]}...```
                        """)
                else:
                    st.warning("No relevant documents found")
                status.update(label="✅ Knowledge lookup completed", state="complete")
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                context["text"] = f"Error retrieving knowledge: {str(e)}"
    
    # Analyze emotions for tone adjustment
    tone_instruction = "Neutral professional tone"
    if st.session_state.sentiment_agent:
        with st.status("💭 Analyzing sentiment...", expanded=True) as status:
            try:
                analysis = st.session_state.sentiment_agent.analyze(user_query)
                tone_instruction = st.session_state.sentiment_agent.generate_tone_instruction(analysis)
                
                # Display sentiment analysis results
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("### Customer Sentiment")
                    if analysis.get("sentiment"):
                        sentiment = analysis["sentiment"]
                        st.metric(
                            label="Dominant Sentiment",
                            value=sentiment["label"].upper(),
                            delta=f"Confidence: {sentiment['score']:.2%}"
                        )
                
                with cols[1]:
                    st.markdown("### Emotional Analysis")
                    if analysis.get("emotions"):
                        for emotion in analysis["emotions"][:3]:
                            st.progress(
                                emotion["score"],
                                text=f"{emotion['label'].title()} ({emotion['score']:.2%})"
                            )
                
                status.update(label="✅ Sentiment analysis completed", state="complete")
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {str(e)}")
                st.error("Sentiment analysis failed")
    
    # Generate AI response with resolution controls
    response_content = ""
    if st.session_state.llm_agent:
        with st.status("💡 Formulating response...") as status:
            try:
                # Generate response content
                response_content = st.session_state.llm_agent.generate_response(
                    query=user_query,
                    context=context["text"],
                    history=st.session_state.session_manager.current_session['history'],
                    tone_instruction=tone_instruction
                )
                
                # Simulate streaming response
                response_placeholder = st.empty()
                full_response = ""
                for chunk in st.session_state.llm_agent.generate_response_stream(
                    query=user_query,
                    context=context["text"],
                    history=st.session_state.session_manager.current_session['history'],
                    tone_instruction=tone_instruction
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Add assistant response to history
                st.session_state.session_manager.add_message_to_current_session(
                    role="assistant",
                    content=full_response
                )
                message_index = len(st.session_state.session_manager.current_session['history']) - 1
                
                # Add resolution controls
                st.markdown("---")
                cols = st.columns([1, 4])
                with cols[0]:
                    if st.button("✅ Mark as Resolved", key=f"resolve_{message_index}"):
                        st.session_state.session_manager.mark_message_resolved(message_index)
                        st.experimental_rerun()
                with cols[1]:
                    st.caption("Was this response helpful?")
                
                status.update(label="✅ Response generated", state="complete")

            except Exception as e:
                response_content = f"Error generating response: {str(e)}"
                logger.error(f"Response generation failed: {str(e)}")
    
    # Process training queue in background
    try:
        st.session_state.session_manager.process_training_queue()
    except Exception as e:
        logger.error(f"Background training error: {str(e)}")
    
    return response_content

# Sidebar components
with st.sidebar:
    st.markdown("""
    <div style='text-align: center;'>
        <h2>🤖 AI Assistant Console</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("**Session Management**")
    
    # Session selection
    current_sessions = st.session_state.session_manager.get_all_session_titles()
    selected_session = st.selectbox(
        "Active Sessions",
        options=["New Session"] + current_sessions
    )
    
    if selected_session == "New Session":
        new_session_name = st.text_input("Session Name:", "Untitled Session")
        if st.button("Create New"):
            st.session_state.session_manager.create_session(new_session_name)
    
    # Data management
    st.markdown("---")
    st.write("**Conversation Tools**")
    if st.button("🧹 Clear Current Session"):
        st.session_state.session_manager.clear_current_session()
        
    st.download_button(
        "📥 Export Conversation",
        data=json.dumps(st.session_state.session_manager.current_session, indent=2),
        file_name="conversation.json",
        mime="application/json"
    )

# Main interface
st.title("🤖 Customer Service AI Assistant")
st.caption("Powered by Advanced Language Understanding")

# Display chat history with resolution controls
for idx, message in enumerate(st.session_state.session_manager.current_session.get('history', [])):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(f"{datetime.fromisoformat(message['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show resolution controls for assistant messages
        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 3])
            with col1:
                if not message.get('resolved'):
                    if st.button("✅ Mark as Resolved", key=f"resolve_{idx}"):
                        st.session_state.session_manager.mark_message_resolved(idx)
            with col2:
                if message.get('resolved'):
                    if 'rating' not in message:
                        display_rating_buttons(idx)
                    else:
                        st.markdown(f"**Your Rating:** {message['rating']} ⭐")

# User input handling
if prompt := st.chat_input("How can I assist you today?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = process_user_input(prompt)
        st.markdown(response)

# Deployment configurations
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

if not os.path.exists("vercel.json"):
    with open("vercel.json", "w") as f:
        json.dump({
            "builds": [{"src": "app.py", "use": "@vercel/python"}],
            "routes": [{"src": "/(.*)", "dest": "app.py"}]
        }, f, indent=2)