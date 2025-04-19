import os
import json
import logging
import uuid
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from collaboration.hitl_manager import HITLManager
from explainability.emotion_lrp import EmotionExplainer
from cultural_awareness.language_detector import LanguageDetector
from cultural_awareness.fairness_audit import BiasAuditor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set environment variables from Streamlit secrets
try:
    os.environ["OPENAI_API_KEY"] = st.secrets.openai.api_key
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
except AttributeError as e:
    logger.error("Missing Streamlit secrets configuration: %s", str(e))
    st.error("🔐 Configuration error: Missing API key setup")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Customer Service Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy imports with error handling
def import_agents():
    try:
        from sentiment_agent import SentimentAgent
        from knowledge_agent import KnowledgeAgent
        from llm_agent import LLMAgent
        from session_manager import SessionManager
        from data_sanitizer import DataSanitizer
        from visualization.emotion_charts import EmotionVisualizer
        from collaboration.hitl_manager import HITLManager
        from explainability.emotion_lrp import EmotionExplainer
        from cultural_awareness.language_detector import LanguageDetector
        from cultural_awareness.fairness_audit import BiasAuditor
        return (SentimentAgent, KnowledgeAgent, LLMAgent, 
                SessionManager, DataSanitizer, EmotionVisualizer, HITLManager,
                EmotionExplainer, LanguageDetector, BiasAuditor)
    except ImportError as e:
        st.error(f"Module import error: {str(e)}")
        st.stop()

(SentimentAgent, KnowledgeAgent, LLMAgent, 
 SessionManager, DataSanitizer, EmotionVisualizer, HITLManager, EmotionExplainer, LanguageDetector, BiasAuditor) = import_agents()

# Initialize session state
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
    st.session_state.current_session_id = None

if 'debug_data' not in st.session_state:
    st.session_state.debug_data = {}

# Agent initialization state
agent_initialization_fields = [
    'knowledge_agent', 'sentiment_agent', 'llm_agent',
    'data_sanitizer', 'hitl_manager', 'cultural_detector',
    'bias_auditor', 'explainer'
]

for field in agent_initialization_fields:
    if field not in st.session_state:
        st.session_state[field] = None

# Initialize agents with caching
@st.cache_resource
def initialize_agents():
    try:
        session_manager = SessionManager()
        visualizer = EmotionVisualizer(session_manager)
        
        return (
            KnowledgeAgent(),
            SentimentAgent(),
            LLMAgent(model_name="gpt-4-turbo-preview"),
            session_manager,
            visualizer
        )
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        st.error("System initialization failed. Please check logs.")
        st.stop()

def initialize_system():
    if None in [st.session_state[field] for field in agent_initialization_fields]:
        with st.spinner("🚀 Loading AI components..."):
            try:
                knowledge_agent, sentiment_agent, llm_agent, session_manager, visualizer = initialize_agents()
                
                # Initialize all agents
                st.session_state.update({
                    'knowledge_agent': knowledge_agent,
                    'sentiment_agent': sentiment_agent,
                    'llm_agent': llm_agent,
                    'session_manager': session_manager,
                    'visualizer': visualizer,
                    'data_sanitizer': DataSanitizer(),
                    'hitl_manager': HITLManager(session_manager),
                    'cultural_detector': LanguageDetector(),
                    'bias_auditor': BiasAuditor(),
                    'explainer': EmotionExplainer(
                        sentiment_agent.emotion_classifier.model,
                        sentiment_agent.emotion_classifier.tokenizer
                    )
                })
                
                # Create initial session if none exists
                if not st.session_state.session_manager.sessions:
                    new_session = st.session_state.session_manager.create_session("Initial Session")
                    st.session_state.current_session_id = new_session['id']  # Direct ID access
                    
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                st.stop()

initialize_system()

def get_current_session():
    return st.session_state.session_manager.get_session(
        st.session_state.current_session_id
    )

def create_new_session(session_name: str):
    new_session = st.session_state.session_manager.create_session(session_name)
    st.session_state.current_session_id = new_session['id']
    st.rerun()

def switch_session(session_id: str):
    st.session_state.current_session_id = session_id
    st.rerun()

def display_rating_buttons(session_id: str, message_index: int):
    session = st.session_state.session_manager.get_session(session_id)
    current_message = session['history'][message_index]
    
    st.markdown("---")
    cols = st.columns([2, 3])
    with cols[0]:
        if not current_message.get('resolved'):
            if st.button(
                "✅ Mark Resolved",
                key=f"resolve_{session_id}_{message_index}",
                help="Confirm this response solved your issue",
                type="primary"
            ):
                st.session_state.session_manager.mark_message_resolved(session_id, message_index)
                st.rerun()
    with cols[1]:
        if current_message.get('resolved'):
            if current_message.get('rating'):
                st.success(f"Rated: {'⭐' * current_message['rating']}")
            else:
                st.write("**Rate the solution quality:**")
                rating_cols = st.columns(5)
                for i in range(1, 6):
                    with rating_cols[i-1]:
                        if st.button(
                            f"{i}⭐",
                            key=f"rate_{session_id}_{message_index}_{i}",
                            on_click=lambda sid=session_id, mi=message_index, r=i: 
                                st.session_state.session_manager.mark_message_resolved(sid, mi, r)
                        ):
                            st.rerun()

def process_user_input(user_query: str, session_id: str):
    try:
        # Cross-cultural analysis
        lang_info = st.session_state.cultural_detector.detect_language(user_query)
        sanitized_query = st.session_state.data_sanitizer.sanitize_text(user_query)
        
        # Sentiment analysis
        analysis = st.session_state.sentiment_agent.analyze(sanitized_query)
        tone_guidance = st.session_state.sentiment_agent.generate_tone_guidance(analysis)

        if 'explainer' not in st.session_state:
            st.session_state.explainer = EmotionExplainer(model, tokenizer)
            
        # Bias auditing
        if st.session_state.bias_auditor:
            bias_report = st.session_state.bias_auditor.audit(
                predictions=[analysis], 
                ground_truth=[]
            )
            st.session_state.debug_data['bias_report'] = bias_report

        # HITL escalation check
        if st.session_state.hitl_manager.check_escalation_needed():
            return _handle_escalation()
        

        # Explanation generation
        explanation = st.session_state.explainer.explain(user_query)
        st.session_state.visualizer.display_explanations(explanation)

        # Convert tone guidance to instruction string
        tone_instruction = f"""
        Respond with:
        - Base tone: {tone_guidance['base_tone']}
        - Strategy: {tone_guidance['emotional_strategy'].get('structure', 'general')}
        - Empathy level: {tone_guidance['emotional_strategy'].get('empathy', 2)}/5
        - Urgency: {tone_guidance['urgency_level']}
        """

        # Emotion timeline update
        timeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": analysis['sentiment']['score'],
            "valence": analysis['valence'],
            "dominant_emotion": analysis['emotions'][0]['label'] if analysis['emotions'] else 'neutral',
            "intensity_trend": analysis['intensity_trend']
        }        
        st.session_state.debug_data = {
            'last_analysis': analysis,
            'timeline_entry': timeline_entry,
            'current_timeline': st.session_state.session_manager.current_session['emotion_timeline']
        }
        current_session = st.session_state.session_manager.get_session(session_id)
        current_session['emotion_timeline'].append(timeline_entry)
        
        # Store message with analytics
        st.session_state.session_manager.add_message_to_session(
            session_id=session_id,
            role="user",
            content=sanitized_query,
            sentiment_score=analysis['sentiment']['score'],
            emotions=analysis['emotions']
        )

        # Knowledge retrieval
        context = {"text": "", "sources": []}
        response_container = st.empty()
        
        if st.session_state.knowledge_agent:
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    context = st.session_state.knowledge_agent.get_context(sanitized_query)
                    if context.get("sources"):
                        with response_container.container():
                            with st.status("🔍 Analyzing knowledge base...", expanded=True) as status:
                                st.markdown("### Relevant Information Found")
                                for idx, source in enumerate(context["sources"][:3]):
                                    cols = st.columns([1, 10])
                                    with cols[0]:
                                        st.success("✅")
                                    with cols[1]:
                                        st.markdown(f"""
                                        **Match {idx+1}**  
                                        **Relevance:** `{source.get('score', 0.0):.2f}`  
                                        **Excerpt:**  
                                        ```{source.get('content', 'No content available')[:200]}...```
                                        """)
                                    st.markdown("---")
                                status.update(label=f"✅ Found {len(context['sources'])} relevant items", state="complete")
                    else:
                        with st.status("🔍 Knowledge Search", expanded=True) as status:
                            st.warning("No relevant documents found")
                            status.update(label="⚠️ No matches found", state="error")
                except Exception as e:
                    logger.error(f"Knowledge retrieval error: {str(e)}")
                    context["text"] = f"Error: {str(e)}"

        # Generate tone instruction
                try:
                    tone_guidance = st.session_state.sentiment_agent.generate_tone_guidance(analysis)
                except KeyError as e:
                    logger.error(f"Tone guidance error: {str(e)}")
                    tone_instruction = f"""
                    Respond using:
                    - Base tone: {tone_guidance['base_tone']}
                    - Strategy: {tone_guidance['emotional_strategy']['structure']}
                    - Empathy: {tone_guidance['emotional_strategy']['empathy']}/5
                    - Urgency: {tone_guidance['urgency_level']}
                    """
                
                # Display sentiment analysis
                with st.status("💭 Analyzing emotions...", expanded=True) as status:
                    try:
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown("### Customer Sentiment")
                            sentiment = analysis["sentiment"]
                            st.metric(
                                label="Dominant Mood",
                                value=sentiment["label"].upper(),
                                delta=f"Confidence: {sentiment['score']:.2%}"
                            )
                        with cols[1]:
                            st.markdown("### Emotional Breakdown")
                            if analysis["emotions"]:
                                for emotion in analysis["emotions"][:3]:  # Show top 3 emotions
                                    st.markdown(
                                        f"**{emotion['label'].title()}**  \n"
                                        f"`{emotion['score']:.2%}` confidence"
                                    )
                                    st.progress(
                                        emotion["score"],
                                        text=f"Intensity Level: {emotion['score']:.0%}"
                                    )
                            else:
                                st.warning("No strong emotions detected")
                        status.update(label="✅ Emotional analysis completed", state="complete")
                    except Exception as e:
                        logger.error(f"Sentiment display error: {str(e)}")
                        st.error("Failed to display emotion analysis")

        # Response generation
        response_content = ""
        if st.session_state.llm_agent:
            response_container = st.empty()
            displayed_response = ""
            
            try:
                for chunk in st.session_state.llm_agent.generate_response_stream(
                    query=sanitized_query,
                    context={
                        "text": context['text'],
                        "sentiment_analysis": {
                            'label': analysis['sentiment']['label'],
                            'intensity': analysis['sentiment']['score'],
                            'valence': analysis['valence']
                        }
                    },
                    history=st.session_state.session_manager.current_session['history'],
                    tone_instruction=tone_instruction
                ):
                    displayed_response += chunk
                    response_container.markdown(displayed_response + "▌")
                    
                response_container.markdown(displayed_response)
                st.session_state.session_manager.add_message_to_current_session(
                    role="assistant",
                    content=displayed_response
                )
                
                # Resolution controls
                message_index = len(st.session_state.session_manager.current_session['history']) - 1
                st.markdown("---")
                cols = st.columns([1, 4])
                with cols[0]:
                    if st.button("✅ Mark Resolved", 
                            key=f"resolve_{message_index}",
                            help="Mark this response as finalized"):
                        st.session_state.session_manager.mark_message_resolved(message_index)
                        st.rerun()
                with cols[1]:
                    if st.session_state.session_manager.current_session['history'][message_index].get('resolved'):
                        display_rating_buttons(message_index)
                    else:
                        st.caption("Rate resolution after marking resolved")
                        
            except Exception as e:
                response_content = f"⚠️ Error: {str(e)}"
                logger.error(f"Response generation failed: {str(e)}")

        return response_content
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return f"System error: {str(e)}"

# Sidebar components
def sidebar_interface():
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center;'>
            <h2>📊 Analytics Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Session Management
        st.write("### Session Management")
        
        # Create new session
        new_name = st.text_input("New Session Name:", "Untitled Conversation")
        if st.button("➕ Create New Session"):
            create_new_session(new_name)
        
        # List existing sessions
        st.write("#### Active Sessions")
        for session in st.session_state.session_manager.sessions:
            cols = st.columns([3, 1])
            with cols[0]:
                st.write(f"**{session['title']}**")
                st.caption(f"Created: {datetime.fromisoformat(session['created_at']).strftime('%Y-%m-%d %H:%M')}")
            with cols[1]:
                if st.button(
                    "🔁",
                    key=f"switch_{session['id']}",
                    help="Switch to this session",
                    on_click=switch_session,
                    args=(session['id'],)
                ):
                    pass
        
        st.markdown("---")
        if st.checkbox("📈 Show Emotion Analytics"):
            current_session = get_current_session()
            st.session_state.visualizer.display_analytics_dashboard(current_session)
            
        # Session tools
        if st.button("🧹 Clear Current Session"):
            st.session_state.session_manager.clear_session(st.session_state.current_session_id)
            st.rerun()
            
        st.download_button(
            "📥 Export Conversation",
            data=json.dumps(get_current_session(), indent=2),
            file_name="conversation.json",
            mime="application/json"
        )
        
        # Debug tools
        st.markdown("---")
        if st.checkbox("Show debug data"):
            st.write("### Debug Information")
            st.write(st.session_state.debug_data)
            
        if st.checkbox("📈 Show Emotion Analytics"):
            st.session_state.visualizer.display_analytics_dashboard(get_current_session())

def main_interface():
    st.title("🤖 AI Customer Support Assistant")
    st.caption("Enhanced with Emotional Intelligence and Context Awareness")
    
    current_session = get_current_session()
    
    # Display chat history
    for idx, message in enumerate(current_session.get('history', [])):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(f"{datetime.fromisoformat(message['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            if message["role"] == "assistant":
                display_rating_buttons(current_session['id'], idx)
    
    # Input handling
    if prompt := st.chat_input("How can I help you today?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = process_user_input(prompt, current_session['id'])
            if response:
                st.markdown(response)

# Run the app
sidebar_interface()
main_interface()

# Deployment configuration
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("""
streamlit==1.30.0
openai==1.14.0
langchain==0.1.9
langchain-openai==0.0.8
langchain-community==0.0.25
faiss-cpu==1.7.4
python-dotenv==1.0.0
transformers==4.35.2
torch==2.1.0
numpy==1.26.2
sympy==1.12.0
plotly==5.18.0
pandas==2.1.4
presidio-analyzer==2.2.33
presidio-anonymizer==2.2.33
reportlab==4.0.4
sentence-transformers==2.2.2
spacy==3.7.0
typing-extensions==4.12.0
huggingface_hub==0.16.4
httpx==0.27.2
onnxruntime==1.16.3
langdetect==1.0.9
regex==2023.12.25
captum==0.7.0
fairlearn==0.8.0
""")

if not os.path.exists("vercel.json"):
    with open("vercel.json", "w") as f:
        json.dump({
            "builds": [{"src": "app.py", "use": "@vercel/python"}],
            "routes": [{"src": "/(.*)", "dest": "app.py"}]
        }, f, indent=2)