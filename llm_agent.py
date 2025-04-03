import os
from typing import List, Dict, Any
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

class LLMAgent:
    """
    Agent responsible for generating responses using a language model.
    Handles system prompts, conversation history and response generation.
    """
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview", temperature: float = 0.7):
        """
        Initialize the LLM agent with a specific model.
        
        Args:
            model_name (str): The name of the OpenAI model to use
            temperature (float): Temperature parameter for generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self.chat_model = self._initialize_chat_model()
        
    def _initialize_chat_model(self):
        """Initialize the chat model using LangChain"""
        return ChatOpenAI(
            temperature=self.temperature,
            model=self.model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def generate_response(self, query: str, context: str, history: List[Dict[str, Any]], 
                         tone_instruction: str = "") -> str:
        """
        Generate a response using the LLM based on query, context, and conversation history.
        
        Args:
            query (str): The user's query
            context (str): Relevant context from knowledge base
            history (List[Dict]): Conversation history
            tone_instruction (str): Optional instruction for response tone
            
        Returns:
            str: Generated response
        """
        try:
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

            # Add current query with tone instruction if provided
            query_with_tone = f"{tone_instruction} {query}" if tone_instruction else query
            messages.append(HumanMessage(content=query_with_tone))

            # Call ChatOpenAI
            response = self.chat_model.invoke(messages)

            return response.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
