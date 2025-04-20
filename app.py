import os
import json
import logging
import uuid
from datetime import datetime
import streamlit as st
import logging
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional audio recorder
try:
    from streamlit_audiorecorder import audiorecorder
except ImportError:
    logging.getLogger(__name__).warning(
        "streamlit_audiorecorder not installed; disabling voice input"
    )
    def audiorecorder(start_label, stop_label, key=None):
        return None

# Optional audio recorder
try:
    from streamlit_audiorecorder import audiorecorder
except ImportError:
    logging.getLogger(__name__).warning(
        "streamlit_audiorecorder not installed; disabling voice input"
    )
    def audiorecorder(start_label, stop_label, key=None):
        return None

# ElevenLabs TTS (optional)
try:
    from elevenlabs import generate, set_api_key
except ImportError:
    logger.warning("elevenlabs package not installed; disabling TTS")
    def set_api_key(key):
        pass
    def generate(text, voice, model):
        return None


# Internal components
from collaboration.hitl_manager import HITLManager
from explainability.emotion_lrp import EmotionExplainer
from cultural_awareness.language_detector import LanguageDetector
from cultural_awareness.fairness_audit import BiasAuditor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
def load_env():
    load_dotenv()
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets.openai.api_key
        eleven_key = st.secrets.elevenlabs.api_key
        set_api_key(eleven_key)
        st.session_state.eleven_key = eleven_key
    except Exception as e:
        logger.error(f"API configuration error: {e}")
        st.error("🔐 Missing API key configuration")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 AI Customer Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize environment
load_env()

# Lazy imports for agents
@st.cache_resource
def import_agents():
    from sentiment_agent import SentimentAgent
    from knowledge_agent import KnowledgeAgent
    from llm_agent import LLMAgent
    from session_manager import SessionManager
    from data_sanitizer import DataSanitizer
    from visualization.emotion_charts import EmotionVisualizer
    return SentimentAgent, KnowledgeAgent, LLMAgent, SessionManager, DataSanitizer, EmotionVisualizer

# Unpack agent classes
SentimentAgent, KnowledgeAgent, LLMAgent, SessionManager, DataSanitizer, EmotionVisualizer = import_agents()

# Initialize session and system state
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

# Initialize system components
@st.cache_resource
def initialize_system():
    session_mgr = st.session_state.session_manager
    visualizer = EmotionVisualizer(session_mgr)
    return (
        KnowledgeAgent(),
        SentimentAgent(),
        LLMAgent(model_name="gpt-4-turbo-preview"),
        DataSanitizer(),
        HITLManager(session_mgr),
        LanguageDetector(),
        BiasAuditor(),
        EmotionExplainer,  # class reference
        visualizer
    )

(
    knowledge_agent,
    sentiment_agent,
    llm_agent,
    data_sanitizer,
    hitl_manager,
    cultural_detector,
    bias_auditor,
    ExplainerClass,
    visualizer
) = initialize_system()

# Instantiate explainer
explainer = ExplainerClass(
    sentiment_agent.emotion_classifier.model,
    sentiment_agent.emotion_classifier.tokenizer
)

# Session management helpers
def get_current_session():
    if not st.session_state.current_session_id:
        new_sess = st.session_state.session_manager.create_session("Initial Session")
        st.session_state.current_session_id = new_sess['id']
        return new_sess
    return st.session_state.session_manager.get_session(st.session_state.current_session_id)

def create_new_session(name: str):
    new_sess = st.session_state.session_manager.create_session(name)
    st.session_state.current_session_id = new_sess['id']
    st.experimental_rerun()

def switch_session(sess_id: str):
    st.session_state.current_session_id = sess_id
    st.experimental_rerun()

def display_rating_buttons(session_id: str, message_index: int):
    session = st.session_state.session_manager.get_session(session_id)
    msg = session['history'][message_index]
    st.markdown("---")
    cols = st.columns([2, 3])
    with cols[0]:
        if not msg.get('resolved'):
            if st.button("✅ Mark Resolved", key=f"resolve_{session_id}_{message_index}"):
                st.session_state.session_manager.mark_message_resolved(session_id, message_index)
    with cols[1]:
        if msg.get('resolved'):
            if msg.get('rating'):
                st.success(f"Rated: {'⭐' * msg['rating']}")
            else:
                st.write("**Rate the solution quality:**")
                stars = st.columns(5)
                for i in range(1, 6):
                    with stars[i-1]:
                        if st.button(f"{i}⭐", key=f"rate_{session_id}_{message_index}_{i}"):
                            st.session_state.session_manager.mark_message_resolved(session_id, message_index, i)
                            st.experimental_rerun()

# Human escalation helper
def _handle_escalation():
    sess = get_current_session()
    escalation_msg = "🚨 Transferring to human agent... Please wait."
    st.session_state.session_manager.add_message_to_session(
        sess['id'], 'system', escalation_msg
    )
    hitl_manager.trigger_human_intervention(sess['id'], reason="Negative sentiment escalation")
    return escalation_msg

# Core processing function
def process_user_input(text, audio, image_file, voice_option, session_id):
    try:
        # Cross-cultural analysis & sanitization
        lang_info = cultural_detector.detect_language(text)
        sanitized_query = data_sanitizer.sanitize_text(text)

        # Multimodal sentiment analysis & tone guidance
        analysis = sentiment_agent.analyze(
            text=sanitized_query,
            audio=audio,
            image=image_file
        )
        tone_guidance = sentiment_agent.generate_tone_guidance(analysis)

        # Bias auditing
        bias_report = bias_auditor.audit(predictions=[analysis], ground_truth=[])
        st.session_state.debug_data = {'bias_report': bias_report}

        # HITL escalation
        if hitl_manager.check_escalation_needed(session_id):
            return _handle_escalation()

        # Explanation generation
        explanation = explainer.explain(sanitized_query)
        visualizer.display_explanations(explanation)

        # Tone instruction for LLM
        tone_instruction = (
            f"""Respond with:\n"
            f"- Base tone: {tone_guidance['base_tone']}\n"
            f"- Strategy: {tone_guidance['emotional_strategy'].get('structure','general')}\n"
            f"- Empathy: {tone_guidance['emotional_strategy'].get('empathy',2)}/5\n"
            f"- Urgency: {tone_guidance['urgency_level']}\n"""
        )

        # Emotion timeline update
        timeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": analysis['sentiment']['score'],
            "valence": analysis['valence'],
            "dominant_emotion": analysis['emotions'][0]['label'] if analysis['emotions'] else 'neutral',
            "intensity_trend": analysis.get('intensity_trend')
        }
        st.session_state.debug_data.update({
            'last_analysis': analysis,
            'timeline_entry': timeline_entry,
            'current_timeline': get_current_session()['emotion_timeline']
        })
        session = st.session_state.session_manager.get_session(session_id)
        session['emotion_timeline'].append(timeline_entry)

        # Store user message
        st.session_state.session_manager.add_message_to_session(
            session_id=session_id,
            role="user",
            content=sanitized_query,
            sentiment_score=analysis['sentiment']['score'],
            emotions=analysis['emotions']
        )

        # Knowledge retrieval
        context = {"text": "", "sources": []}
        kb_container = st.empty()
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                context = knowledge_agent.get_context(sanitized_query)
                if context.get("sources"):
                    with kb_container.container():
                        st.markdown("### Relevant Information Found")
                        for idx, src in enumerate(context["sources"][:3]):
                            cols = st.columns([1, 10])
                            with cols[0]: st.success("✅")
                            with cols[1]:
                                st.markdown(
                                    f"**Match {idx+1}**  \n"
                                    f"**Relevance:** `{src.get('score',0):.2f}`  \n"
                                    f"```{src.get('content','')[:200]}...```"
                                )
                            st.markdown("---")
                else:
                    st.warning("No relevant documents found")
            except Exception as e:
                logger.error(f"Knowledge retrieval error: {e}")
                context["text"] = f"Error: {e}"

        # Display sentiment metrics
        with st.expander("💭 Customer Sentiment"):
            cols = st.columns(2)
            with cols[0]:
                st.metric(
                    label="Dominant Mood",
                    value=analysis['sentiment']['label'].upper(),
                    delta=f"Confidence: {analysis['sentiment']['score']:.2%}"
                )
            with cols[1]:
                st.markdown("### Emotional Breakdown")
                for emo in analysis.get('emotions', [])[:3]:
                    st.markdown(f"**{emo['label'].title()}**  `{emo['score']:.2%}`")
                    st.progress(emo['score'], text=f"{emo['score']:.0%}")

        # LLM response streaming
        response = ""
        stream_container = st.empty()
        for chunk in llm_agent.generate_response_stream(
            query=sanitized_query,
            context={"text": context['text'], "sentiment_analysis": analysis},
            history=session['history'],
            tone_instruction=tone_instruction
        ):
            response += chunk
            stream_container.markdown(response + "▌")
        stream_container.markdown(response)

        # Store assistant response
        st.session_state.session_manager.add_message_to_session(
            session_id=session_id,
            role="assistant",
            content=response
        )

        # Text-to-speech reply
        if st.session_state.eleven_key:
            audio_resp = generate(
                text=response,
                voice=voice_option,
                model="eleven_multilingual_v1"
            )
            st.session_state.session_manager.add_message_to_session(
                session_id=session_id,
                role="assistant",
                content=audio_resp,
                content_type="audio"
            )

        return response
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return f"⚠️ Error during processing: {e}"

# Sidebar UI
def sidebar_interface():
    with st.sidebar:
        st.header("🎛️ Controls")
        audio = audiorecorder("Record", "Stop", key="recorder")
        if audio:
            st.audio(audio, format="audio/wav")
        st.session_state.audio_data = audio

        st.subheader("🖼️ Image Input")
        img = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
        if img:
            st.image(img, use_column_width=True)
        st.session_state.uploaded_image = img

        voice = st.selectbox(
            "Assistant Voice",
            ["alloy","echo","fable","onyx","nova","shimmer"],
            index=0
        )
        st.session_state.voice_option = voice

        st.markdown("---")
        if st.button("🧹 Clear Conversation"):
            st.session_state.session_manager.clear_session(st.session_state.current_session_id)
            st.experimental_rerun()

        st.markdown("---")
        name = st.text_input("New Session Name:", "Untitled Conversation")
        if st.button("➕ Create New Session"):
            create_new_session(name)

        current = get_current_session()
        sessions = st.session_state.session_manager.list_sessions()
        for sess in sessions[:3]:
            cols = st.columns([3,1])
            with cols[0]:
                st.write(f"**{sess['title']}**")
                st.caption(f"Created: {datetime.fromisoformat(sess['created']).strftime('%Y-%m-%d %H:%M')}")
            with cols[1]:
                if st.button("🔁", key=sess['id']):
                    switch_session(sess['id'])
        if len(sessions) > 3:
            if st.button("📜 Show All Sessions"):
                for sess in sessions:
                    cols = st.columns([3,1])
                    with cols[0]: st.write(f"**{sess['title']}**")
                    with cols[1]:
                        if st.button("🔁", key=f"all_{sess['id']}"):
                            switch_session(sess['id'])

        st.markdown("---")
        if st.button("📥 Export Conversation"):
            st.download_button(
                "Download JSON",
                json.dumps(get_current_session(), indent=2),
                file_name="conversation.json",
                mime="application/json"
            )

# Main chat UI
def main_interface():
    st.title("🤖 AI Customer Support Assistant")
    st.caption("Enhanced with Emotional Intel & Multimodal I/O")

    current = get_current_session()
    for idx, msg in enumerate(current['history']):
        with st.chat_message(msg['role']):
            if msg.get('content_type') == 'audio':
                st.audio(msg['content'], format="audio/wav")
            elif msg.get('content_type') == 'image':
                st.image(msg['content'], use_column_width=True)
            else:
                st.markdown(msg['content'])
            st.caption(datetime.fromisoformat(msg['timestamp']).strftime('%Y-%m-%d %H:%M:%S'))
            if msg['role'] == 'assistant':
                display_rating_buttons(current['id'], idx)

    prompt = st.chat_input("Type your message...")
    if prompt or st.session_state.audio_data or st.session_state.uploaded_image:
        with st.spinner("Analyzing..."):
            process_user_input(
                text=prompt,
                audio=st.session_state.audio_data,
                image_file=st.session_state.uploaded_image,
                voice_option=st.session_state.voice_option,
                session_id=current['id']
            )
        st.experimental_rerun()

# Entry point
if __name__ == "__main__":
    sidebar_interface()
    main_interface()

# Deployment artifacts
if not os.path.exists("requirements.txt"):
    with open("requirements.txt","w") as f:
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
uuid==1.30
""")

if not os.path.exists("vercel.json"):
    with open("vercel.json","w") as f:
        json.dump({
            "builds": [{"src": "app.py", "use": "@vercel/python"}],
            "routes": [{"src": "/(.*)", "dest": "app.py"}]
        }, f, indent=2)
