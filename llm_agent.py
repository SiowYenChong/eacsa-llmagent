import os
import re
import streamlit as st
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI 
import numpy as np
from langchain.schema import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

class LLMAgent:
    """Secure LLM agent with verification enforcement"""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.chat_model = self._initialize_chat_model()
        self.rate_limit = {}

    def generate_response(self, query: str, context: dict, history: list) -> str:
        """Sentiment-aware response generation"""
        sentiment = context.get('sentiment_analysis', {})
        tone_guidance = self.sentiment_agent.generate_tone_guidance(sentiment)
        
        messages = [
            SystemMessage(content=self._build_system_prompt(sentiment, tone_guidance)),
            *self._format_history(history),
            HumanMessage(content=query)
        ]

        return self._generate_with_fallback(messages, sentiment)

    def _build_system_prompt(self, sentiment: dict, tone: dict) -> str:
        """Dynamic prompt engineering based on sentiment"""
        base_prompt = f"""You are an emotionally intelligent customer support agent. 
        Current user sentiment: {sentiment.get('label', 'neutral')} (confidence: {sentiment.get('score', 0):.2f})
        Detected emotions: {', '.join([e['label'] for e in sentiment.get('emotions', [])])}
        Response guidelines:
        - Structure: {tone['structure']}
        - Empathy level: {tone['empathy_level']}/5
        - Formality: {tone['language_formality']*100}%
        - Urgency: {tone['urgency']}
        """
        
        if sentiment.get('context_shift'):
            base_prompt += "\nWARNING: User sentiment has shifted negatively - prioritize de-escalation!"
            
        return base_prompt + "\n\nContext:\n" + context['text']

    def _generate_with_fallback(self, messages: list, sentiment: dict) -> str:
        """Generate response with sentiment-aware retries"""
        for attempt in range(3):
            try:
                response = self.chat_model.invoke(messages).content
                if self._validate_response_quality(response, sentiment):
                    return response
                messages.append(HumanMessage(content="Please rephrase that response to be more empathetic"))
            except Exception as e:
                logger.error(f"Generation attempt {attempt+1} failed: {str(e)}")
        return "I'm having trouble formulating the best response. Let me connect you with a human specialist."

    def _get_similar_query_rating(self, query: str, context: dict) -> float:
        """Get average rating for similar historical responses"""
        similar_cases = self.session_manager.find_similar_queries(
            query=query,
            sentiment=context.get('sentiment_label'),
            product_ids=context.get('products', [])
        )
        return np.mean([case['rating'] for case in similar_cases]) if similar_cases else 3.0

    def _apply_rating_insights(self, response: str, query: str, 
                            sentiment: dict, similarity_score: float) -> str:
        """Modify response based on historical rating patterns"""
        optimization_rules = {
            'negative': [
                (lambda: similarity_score < 2.5, "Add apology and urgency"),
                (lambda: 'frustration' in sentiment['emotions'], "Use empathetic framing")
            ],
            'positive': [
                (lambda: similarity_score > 4.0, "Reinforce positive language"),
                (lambda: 'excitement' in sentiment['emotions'], "Add celebratory tone")
            ]
        }

        for condition, action in optimization_rules.get(sentiment['label'], []):
            if condition():
                response = self._modify_response(response, action)
        
        return response

    def _modify_response(self, response: str, action: str) -> str:
        """Apply specific response modifications"""
        modifications = {
            "Add apology and urgency": lambda x: f"I apologize for the inconvenience. {x} Let's resolve this immediately!",
            "Use empathetic framing": lambda x: x.replace("I understand", "I completely understand how frustrating"),
            "Reinforce positive language": lambda x: x + " We're thrilled to have delivered great service!",
            "Add celebratory tone": lambda x: x.replace("Thank you", "🎉 Wow! Thank you")
        }
        return modifications.get(action, lambda x: x)(response)
    
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
            streaming=True,  # Enable streaming
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
        
    def generate_response_stream(self, query: str, context: str, 
                                history: list, tone_instruction: str):
        """Stream generated response chunks"""
        try:
            messages = self._format_messages(query, context, history, tone_instruction)
            
            for chunk in self.chat_model.stream(messages):
                content = chunk.content
                if content is not None:
                    yield content
        except Exception as e:
            yield f"Error: {str(e)}"

    def _format_messages(self, query: str, context: str, 
                        history: list, tone_instruction: str):
        """Helper method to format messages for streaming"""
        messages = [
            SystemMessage(content=f"""
            Context: {context}
            Tone instructions: {tone_instruction}
            """)
        ]
        
        for message in history[-6:]:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
                
        messages.append(HumanMessage(content=query))
        return messages
