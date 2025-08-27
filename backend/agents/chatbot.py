# chatbot.py (input: user_question + rag_system, output: answers)
import logging
from typing import List, Dict, Optional
from lightrag import LightRAG, QueryParam
from agents.filtering import Email

class ChatbotAgent:
    """Agent for answering questions about emails stored in the knowledge graph."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_prompt = """You are an expert email assistant for a computer science engineer. 
        Your task is to help them find information in their important emails. 
        
        You have access to their email knowledge graph which contains:
        - Email content (subjects, senders, bodies)
        - Extracted entities (people, projects, dates, etc.)
        - Relationships between emails and entities
        
        Be concise, professional, and focus on answering based on the email data.
        If you don't find relevant information in the emails, say so.
        """
    
    async def answer_user_question(self, rag: LightRAG, question: str) -> str:
        """
        Answer user questions using the RAG system with email knowledge graph.
        
        Args:
            rag: Initialized LightRAG instance with email data
            question: User's question about their emails
            
        Returns:
            str: Generated answer to the user's question
        """
        try:
            # Use RAG to query the knowledge graph
            response = await rag.generate(
                query=question,
                system_prompt=self.system_prompt,
                query_param=QueryParam(mode="mix")
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error answering user question: {e}")
            return "I encountered an error processing your question. Please try again."