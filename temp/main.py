import os
import json
import time
from datetime import datetime

import openai
import streamlit as st
from sympy.simplify.simplify import bottom_up
from transformers import pipeline
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




# Load both models
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
emotion_classifier = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student",
                              top_k=None)


def analyze_sentiment_and_emotion(user_input):
    """
    Analyzes both the sentiment (positive, neutral, negative) and dominant emotion of the user's input.

    Returns:
        sentiment (str): One of 'negative', 'neutral', 'positive'
        dominant_emotion (str): The most prominent detected emotion
    """

    # Perform sentiment analysis using BERT
    sentiment_result = sentiment_analyzer(user_input)[0]
    sentiment_label = sentiment_result["label"]
    # print(sentiment_label)

    # Map sentiment labels from BERT to categories
    sentiment_map = {
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive"
    }
    sentiment = sentiment_map.get(sentiment_label, "neutral")

    # Perform emotion analysis using GoEmotions BERT
    emotion_results = emotion_classifier(user_input)
    # print(emotion_results,'\n\n\n\n')

    # Ensure we are working with a list of emotions
    if emotion_results and isinstance(emotion_results, list):
        # Sort the emotions by score in descending order
        sorted_emotions = sorted(emotion_results[0], key=lambda x: x.get("score", 0), reverse=True)

        # Get the most confident emotion
        dominant_emotion = sorted_emotions[0].get("label", "neutral")  # Default to neutral if no label found
    else:
        dominant_emotion = "neutral"  # Default if no emotion detected

    return sentiment, dominant_emotion




#
# user_input = "I'm really disappointed with the service, but I hope things improve soon."
# sentiment, emotion = analyze_sentiment_and_emotion(user_input)
# print(f"Sentiment: {sentiment}, Dominant Emotion: {emotion}")






# ====== frontend =======

# Page configuration
st.set_page_config(
    page_title="Customer Service Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sessions' not in st.session_state:
    st.session_state.sessions = []
if 'current_session' not in st.session_state:
    st.session_state.current_session = {
        'id': datetime.now().strftime("%Y%m%d%H%M%S"),
        'title': 'New Session',
        'history': [],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'openai_model' not in st.session_state:
    st.session_state.openai_model = None


# Initialize the chat model
@st.cache_resource
def initialize_chat_model():
    return ChatOpenAI(
        temperature=0.7,
        model="gpt-4-turbo-preview",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


# Function to initialize vector store from documents
@st.cache_resource
def initialize_vector_store():
    try:
        # Check if we have a pre-saved vector store
        if os.path.exists("faiss_index"):
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            return vector_store

        # If not, create from scratch
        # Load documents from the knowledge_base directory
        text_loader = DirectoryLoader("knowledge_base/", glob="**/*.txt", loader_cls=TextLoader)
        text_documents = text_loader.load()

        csv_loader = DirectoryLoader("knowledge_base/", glob="**/*.csv", loader_cls=CSVLoader)
        csv_documents = csv_loader.load()

        documents = text_documents + csv_documents

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)

        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)

        # Save for future use
        vector_store.save_local("faiss_index")

        return vector_store
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None


# Initialize the vector store and chat model
if st.session_state.vector_store is None:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.vector_store = initialize_vector_store()

if st.session_state.openai_model is None:
    st.session_state.openai_model = initialize_chat_model()


# RAG function to get relevant context for user query
def get_context(query, vector_store, max_docs=3):
    if vector_store is None:
        return "Knowledge base unavailable."

    relevant_docs = vector_store.similarity_search(query, k=max_docs)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context

#
# # Function to generate a response using ChatOpenAI
# def get_openai_response(query, context, history):
#     try:
#         chat_model = st.session_state.openai_model
#
#         # Format conversation history for LangChain
#         messages = [
#             SystemMessage(content=f"""You are a helpful customer service assistant.
#             Use the following context to answer the customer's question.
#             If you don't know the answer based on the context, say so politely and suggest they contact human support.
#
#             Context: {context}
#             """)
#         ]
#
#         # Add conversation history
#         for message in history[-6:]:  # Last 6 messages for context
#             if message["role"] == "user":
#                 messages.append(HumanMessage(content=message["content"]))
#             elif message["role"] == "assistant":
#                 messages.append(AIMessage(content=message["content"]))
#
#         # Add current query
#         messages.append(HumanMessage(content=query))
#
#         # Call ChatOpenAI
#         response = chat_model.invoke(messages)
#
#         return response.content
#     except Exception as e:
#         return f"Sorry, I encountered an error: {str(e)}"


# Function to generate a response using ChatOpenAI
def get_openai_response(query, context, history):
    try:
        # Detect sentiment and emotion from the user's query
        sentiment, emotion = analyze_sentiment_and_emotion(query)

        # Create an appropriate tone based on sentiment and emotion
        tone = ""
        if sentiment == "negative" or emotion in ["disappointment", "sadness", "anger", "disapproval"]:
            tone = "Please be patient. I'll do my best to help you with your issue. I'm sorry you're feeling this way, and I'll try to assist in resolving things."
        elif sentiment == "positive" or emotion in ["joy", "excitement", "optimism"]:
            tone = "I'm so glad to hear you're happy! How can I assist you today with a smile?"
        else:
            tone = "Let me know how I can assist you. I'm here to help."

        # Incorporate the tone into the system message
        chat_model = st.session_state.openai_model

        # Format conversation history for LangChain
        messages = [
            SystemMessage(content=f"""You are a helpful customer service assistant.
            Use the following context to answer the customer's question. Adjust your tone based on the user's sentiment and emotions:
            - If the sentiment is negative or the user shows frustration or sadness, respond empathetically.
            - If the sentiment is positive or the user expresses happiness, be cheerful and upbeat.
            - Otherwise, maintain a neutral and helpful tone.
            Context: {context}
            """)
        ]

        # Add conversation history
        for message in history[-6:]:  # Last 6 messages for context
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))

        # Add current query with sentiment and emotion-based tone instruction
        messages.append(HumanMessage(content=f"{tone} {query}"))

        # Call ChatOpenAI
        response = chat_model.invoke(messages)

        return response.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"




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
    session_titles = [session['title'] for session in st.session_state.sessions]
    session_titles.append("🆕 New Session")

    selected_session_title = st.selectbox("Select a Session", session_titles)

    if selected_session_title == "🆕 New Session":
        new_session_title = st.text_input("Name your session:", value=f"Session {len(st.session_state.sessions) + 1}")
        if st.button("Create Session"):
            new_session = {
                'id': datetime.now().strftime("%Y%m%d%H%M%S"),
                'title': new_session_title,
                'history': [],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.sessions.append(new_session)
            st.session_state.current_session = new_session
            st.rerun()
    else:
        for session in st.session_state.sessions:
            if session['title'] == selected_session_title:
                st.session_state.current_session = session
                break

    if st.button("🧹 Clear Current Session"):
        st.session_state.current_session['history'] = []
        st.rerun()

# Main interface
st.markdown("""
<h1 style='text-align: center;'>
    🤖 Customer Service Assistant
</h1>
<p style='text-align: center;'>Ask me anything about our products and services</p>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.current_session['history']:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])
        if role == "assistant":
            st.caption(f"*Bot • {message['timestamp']}*")

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to current session history
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.current_session['history'].append(user_message)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get relevant context for the query
    with st.spinner("🔍 Looking up information..."):
        context = get_context(prompt, st.session_state.vector_store)

    # Get assistant's response
    with st.spinner("💭 Thinking..."):
        assistant_message_content = get_openai_response(
            prompt,
            context,
            st.session_state.current_session['history']
        )

        assistant_message = {
            "role": "assistant",
            "content": assistant_message_content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_message["content"])
        st.caption(f"Generated at: {assistant_message['timestamp']}")

    # Add assistant message to history
    st.session_state.current_session['history'].append(assistant_message)

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










