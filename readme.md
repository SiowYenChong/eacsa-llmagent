# Customer Service Chat Bot 🤖💬

This project is a modular, LLM-powered customer support assistant that uses LangChain, OpenAI, and FAISS to provide context-aware, emotionally intelligent responses to customer queries.

---

## 🧠 Overview

The chatbot combines:
- **LLM response generation** using OpenAI's GPT models
- **Context retrieval** from a vectorized knowledge base (via FAISS)
- **Sentiment analysis** to adapt response tone
- **Session management** to maintain multi-turn conversations

The end result is a smart, empathetic assistant that enhances customer support by delivering accurate, personalized, and emotionally appropriate replies.

---

## 📁 Project Structure

```plaintext
customer-service-chat-bot/
│
├── app.py                 # Main entry point, launches Streamlit app
├── llm_agent.py           # Handles prompt formatting & LLM interaction
├── knowledge_agent.py     # Retrieves relevant context using FAISS
├── sentiment_agent.py     # Analyzes tone/sentiment of user inputs
├── session_manager.py     # Manages chat history and user sessions
│
├── faiss_index/           # Stores vectorized index of knowledge base
├── knowledge_base/        # Contains documents and reference data
├── temp/                  # Temporary storage if needed
│
├── requirements.txt       # Project dependencies
├── vercel.json            # Deployment config (e.g. for Vercel)
└── .venv/                 # Virtual environment (not committed)

---

## ⚙️ Setup Instructions

> ✅ Requires Python **3.10+**

### 1. Clone the repository

```bash
git clone <repo-url>
cd customer-service-chat-bot
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows CMD:
```bash
.venv\Scripts\activate.bat
```
Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```
✅ All versions are pinned for compatibility with FAISS, Streamlit, LangChain, and Transformers.

### 5. Configure Environment Variables
Create a .env file in the project root:
```bash
OPENAI_API_KEY=your-openai-api-key-here
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 🧪 Run the App
Launch the chatbot UI using Streamlit:

```bash
streamlit run app.py
```
---

## ✅ Example Output
User:

I'm having trouble with my order.

Bot:

I'm sorry to hear you're facing issues with your order. Could you please share your order number so I can help you better? Here's also a quick guide on tracking your order status...

💡 How This Works
🔍 Contextual info retrieved by knowledge_agent.py using FAISS

🎭 Tone adaptation powered by sentiment_agent.py

🧠 LLM-generated responses from llm_agent.py

🗂️ Chat continuity managed via session_manager.py

---

## 🚀 Deployment
You can deploy the app using Vercel or any cloud platform.
Customize vercel.json as needed for your configuration.

---
## Maintainer
Made with ⚡ by Aishwar & Siow Yen
