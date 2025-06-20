from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.schema import BaseMessage
import config

def get_llm():
    """Get the configured LLM provider"""
    if config.LLM_PROVIDER.lower() == "groq":
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=1024
        )
    else:  # Default to Gemini
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            google_api_key=config.GOOGLE_API_KEY,
            model="gemini-2.0-flash",
            temperature=0.3,
            max_output_tokens=1024
        )

def format_messages_for_llm(messages: list) -> list:
    """Format messages for the LLM"""
    formatted_messages = []
    
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                from langchain.schema import SystemMessage
                formatted_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                from langchain.schema import AIMessage
                formatted_messages.append(AIMessage(content=content))
            else:  # user or any other role
                from langchain.schema import HumanMessage
                formatted_messages.append(HumanMessage(content=content))
        elif isinstance(message, BaseMessage):
            formatted_messages.append(message)
    
    return formatted_messages