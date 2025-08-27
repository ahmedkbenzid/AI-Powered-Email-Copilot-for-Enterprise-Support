# graph.py
import os
import asyncio
import base64
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm import ollama_model_complete
from lightrag.utils import EmbeddingFunc
import requests

# LangChain for semantic chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

class EmailKnowledgeGraph:
    """Knowledge Graph for processing important emails using RAG-Anything"""
    
    def __init__(self, 
                 working_dir: str = "./rag_storage_neo4j",
                 neo4j_uri: str = "neo4j://localhost:7687",
                 neo4j_username: str = "neo4j",
                 neo4j_password: str = "graph2025",
                 llm_model: str = "llama3.2:3b",
                 embedding_model: str = "nomic-embed-text",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize EmailKnowledgeGraph with configurable parameters
        
        Args:
            working_dir: Working directory for RAG storage
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            llm_model: Ollama LLM model name
            embedding_model: Ollama embedding model name
            ollama_host: Ollama server host
        """
        self.working_dir = working_dir
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.rag = None
        self.chunker = self._create_semantic_chunker()
        
        # Ensure working directory exists
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
            
        # Set Neo4j environment variables
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_username
        os.environ["NEO4J_PASSWORD"] = neo4j_password
    
    def _create_semantic_chunker(self):
        """Create a semantic chunker using Ollama embeddings"""
        embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_host
        )
        
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
    
    def _ollama_embed(self, texts: List[str]) -> List[List[float]]:
        """Ollama embedding function for RAG-Anything"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except requests.exceptions.RequestException as e:
                print(f"Error getting embedding for text: {e}")
                # Fallback: return zero vector (768 for nomic-embed-text)
                embeddings.append([0.0] * 768)
        return embeddings
    
    async def initialize(self):
        """Initialize RAG-Anything with Ollama models"""
        try:
            # Create RAG-Anything configuration
            config = RAGAnythingConfig(
                working_dir=self.working_dir,
                enable_image_processing=True,
                enable_table_processing=False,
                enable_equation_processing=False,
                parser="mineru"
            )
            
            # Define Ollama model functions
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return ollama_model_complete(
                    self.llm_model,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    host=self.ollama_host,
                    options={
                        "num_ctx": 32768,
                        "temperature": 0.3,
                    }
                )
            
            # Vision model function
            def vision_model_func(prompt, system_prompt=None, history_messages=[], 
                                image_data=None, messages=None, **kwargs):
                return ollama_model_complete(
                    "llava:7b",  # Using LLaVA for vision tasks
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    host=self.ollama_host,
                    options={
                        "num_ctx": 32768,
                        "temperature": 0.3,
                    }
                )
            
            # Define embedding function
            embedding_func = EmbeddingFunc(
                embedding_dim=768,  # For nomic-embed-text
                max_token_size=8192,
                func=self._ollama_embed,
            )
            
            # Initialize RAG-Anything
            self.rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )
            
            print("✅ RAG-Anything initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG-Anything: {e}")
            raise
    
    def check_email_attachments(self, gmail_service, email_id: str) -> bool:
        """
        Check if an email has attachments
        
        Args:
            gmail_service: Authenticated Gmail service
            email_id: Gmail message ID
            
        Returns:
            bool: True if email has attachments, False otherwise
        """
        try:
            message = gmail_service.users().messages().get(
                userId='me', 
                id=email_id,
                format='full'
            ).execute()
            
            payload = message.get('payload', {})
            parts = payload.get('parts', [])
            
            # Check for attachments
            for part in parts:
                if part.get('filename') and part.get('filename') != '':
                    return True
                # Check nested parts for attachments
                if 'parts' in part:
                    for sub_part in part['parts']:
                        if sub_part.get('filename') and sub_part.get('filename') != '':
                            return True
            return False
            
        except Exception as e:
            print(f"Error checking attachments for email {email_id}: {e}")
            return False
    
    def extract_attachment_content(self, gmail_service, email_id: str) -> List[Dict]:
        """
        Extract content from email attachments
        
        Args:
            gmail_service: Authenticated Gmail service
            email_id: Gmail message ID
            
        Returns:
            List[Dict]: List of attachment content dictionaries
        """
        attachment_contents = []
        
        try:
            message = gmail_service.users().messages().get(
                userId='me', 
                id=email_id,
                format='full'
            ).execute()
            
            payload = message.get('payload', {})
            parts = payload.get('parts', [])
            
            # Process attachments
            for part in parts:
                if part.get('filename') and part.get('filename') != '':
                    attachment_info = self._process_attachment(gmail_service, email_id, part)
                    if attachment_info:
                        attachment_contents.append(attachment_info)
                        
                # Check nested parts
                if 'parts' in part:
                    for sub_part in part['parts']:
                        if sub_part.get('filename') and sub_part.get('filename') != '':
                            attachment_info = self._process_attachment(gmail_service, email_id, sub_part)
                            if attachment_info:
                                attachment_contents.append(attachment_info)
                                
        except Exception as e:
            print(f"Error extracting attachment content: {e}")
            
        return attachment_contents
    
    def _process_attachment(self, gmail_service, email_id: str, part: dict) -> Dict:
        """Process individual attachment part"""
        try:
            attachment_id = part.get('body', {}).get('attachmentId')
            if not attachment_id:
                return {}
                
            attachment = gmail_service.users().messages().attachments().get(
                userId='me',
                messageId=email_id,
                id=attachment_id
            ).execute()
            
            # Decode attachment data
            data = attachment.get('data', '')
            file_data = base64.urlsafe_b64decode(data)
            filename = part.get('filename', 'unknown')
            mime_type = part.get('mimeType', 'unknown')
            
            # Try to extract text content (simplified approach)
            try:
                # For text files
                content = file_data.decode('utf-8')
                return {
                    "type": "text",
                    "content": content,
                    "filename": filename,
                    "mime_type": mime_type
                }
            except UnicodeDecodeError:
                # For binary files, return basic info
                return {
                    "type": "text",
                    "content": f"[Attachment: {filename} ({mime_type})]",
                    "filename": filename,
                    "mime_type": mime_type
                }
                
        except Exception as e:
            print(f"Error processing attachment: {e}")
            return {}
    
    async def process_important_emails(self, important_emails: List[Tuple], gmail_service):
        """
        Process important emails and add to knowledge graph using RAG-Anything
        
        Args:
            important_emails: List of (Email, classification_result) tuples
            gmail_service: Authenticated Gmail service
        """
        if not self.rag:
            raise Exception("RAG-Anything not initialized. Call initialize() first.")
            
        print(f"Processing {len(important_emails)} important emails...")
        
        for i, (email, classification) in enumerate(important_emails):
            print(f"Processing email {i+1}/{len(important_emails)}: {email.subject[:50]}...")
            
            # Prepare content list for RAG-Anything
            content_list = []
            
            # Check for attachments
            has_attachments = self.check_email_attachments(gmail_service, email.id) if email.id else False
            
            if has_attachments:
                print(f"  Email has attachments - using RAG-Anything with attachment content")
                # Extract attachment content
                attachment_contents = self.extract_attachment_content(gmail_service, email.id)
                
                # Add email body as text content
                email_chunks = self.chunker.split_text(email.body)
                for j, chunk in enumerate(email_chunks):
                    content_list.append({
                        "type": "text",
                        "text": chunk,
                        "page_idx": 0,
                        "metadata": {
                            "chunk_index": j,
                            "source": "email_body"
                        }
                    })
                
                # Add attachment content
                for j, attachment in enumerate(attachment_contents):
                    if attachment.get("type") == "text":
                        content_list.append({
                            "type": "text",
                            "text": attachment.get("content", ""),
                            "page_idx": 1,  # Different page for attachments
                            "metadata": {
                                "chunk_index": j,
                                "source": "email_attachment",
                                "filename": attachment.get("filename", "unknown"),
                                "mime_type": attachment.get("mime_type", "unknown")
                            }
                        })
            else:
                print(f"  Email has no attachments - using semantic chunking with RAG-Anything")
                # Process with semantic chunking only
                chunks = self.chunker.split_text(email.body)
                
                # Add each chunk to content list
                for j, chunk in enumerate(chunks):
                    content_list.append({
                        "type": "text",
                        "text": chunk,
                        "page_idx": 0,
                        "metadata": {
                            "chunk_index": j,
                            "source": "email_body"
                        }
                    })
            
            # Insert content list into RAG-Anything
            if content_list:
                try:
                    await self.rag.insert_content_list(
                        content_list=content_list,
                        file_path=f"email_{email.id}_{email.subject[:30]}.txt",
                        doc_id=f"email_{email.id}" if email.id else None,
                        display_stats=True
                    )
                    print(f"  ✅ Added {len(content_list)} content items to knowledge graph")
                except Exception as e:
                    print(f"  ❌ Error inserting content list: {e}")
            else:
                print(f"  ⚠️  No content to add for this email")
    
    async def query_knowledge_graph(self, query: str, mode: str = "hybrid") -> str:
        """
        Query the knowledge graph using RAG-Anything
        
        Args:
            query: Query string
            mode: Query mode (hybrid, vector, kg, mix)
            
        Returns:
            Query response
        """
        if not self.rag:
            raise Exception("RAG-Anything not initialized. Call initialize() first.")
            
        try:
            response = await self.rag.aquery(query, mode=mode)
            return response
        except Exception as e:
            print(f"Error querying knowledge graph: {e}")
            return f"Error: {str(e)}"

# Usage example:
async def main():
    """Example usage"""
    # Initialize the knowledge graph with custom parameters
    kg = EmailKnowledgeGraph(
        working_dir="./rag_storage_neo4j",
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="graph2025",
        llm_model="llama3.2:3b",
        embedding_model="nomic-embed-text",
        ollama_host="http://localhost:11434"
    )
    await kg.initialize()
    
    # This would be called after you have your important emails
    # await kg.process_important_emails(important_emails, gmail_service)

if __name__ == "__main__":
    asyncio.run(main())