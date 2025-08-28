#!/usr/bin/env python3
"""
FastAPI application for email classification and knowledge graph processing
"""
import uvicorn
import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Global instances
gmail_agent = None
classifier = None
knowledge_graph = None
config = None

# Pydantic models for API
class ClassificationResult(BaseModel):
    """Email classification result model"""
    classification: str = Field(description="IMPORTANT or NOT_IMPORTANT")
    topic: str = Field(description="WORK, PERSONAL, or PROMOTIONAL")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    reasoning: str = Field(description="Classification reasoning")
    success: bool = Field(description="Classification success status")

class EmailWithClassification(BaseModel):
    """Email with its classification result"""
    email: Dict[str, Any]
    classification: ClassificationResult

class EmailFetchRequest(BaseModel):
    """Request model for fetching emails"""
    days_back: int = Field(default=7, description="Days to look back for emails")
    max_emails: int = Field(default=20, description="Maximum number of emails to fetch")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for important emails")

class QueryRequest(BaseModel):
    """Request model for knowledge graph queries"""
    query: str = Field(description="Query string")
    mode: str = Field(default="hybrid", description="Query mode: hybrid, vector, kg, mix")

class ProcessingStatus(BaseModel):
    """Processing status model"""
    status: str = Field(description="Processing status")
    message: str = Field(description="Status message")
    total_emails: Optional[int] = None
    important_emails: Optional[int] = None
    processed_emails: Optional[int] = None

class TextClassificationRequest(BaseModel):
    """Request model for text classification"""
    text: str = Field(description="Text to classify")

def load_config() -> Dict:
    """Load configuration from environment variables"""
    GROQ_API_KEY="REDACTED"
    return {
        'groq_api_key': GROQ_API_KEY,
        'gmail_credentials': os.getenv('GMAIL_CREDENTIALS_FILE', 'credentials.json'),
        'gmail_token': os.getenv('GMAIL_TOKEN_FILE', 'token.pickle'),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')),
        # Knowledge Graph Configuration
        'neo4j_uri': os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
        'neo4j_username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'neo4j_password': os.getenv('NEO4J_PASSWORD', 'graph2025'),
        'llm_model': os.getenv('LLM_MODEL', 'llama3.2:3b'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'nomic-embed-text'),
        'ollama_host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
        'rag_working_dir': os.getenv('RAG_WORKING_DIR', './rag_storage_neo4j'),
        'enable_knowledge_graph': os.getenv('ENABLE_KNOWLEDGE_GRAPH', 'true').lower() == 'true'
    }

async def initialize_services():
    """Initialize all services"""
    global gmail_agent, classifier, knowledge_graph, config
    
    config = load_config()
    
    if not config['groq_api_key']:
        raise RuntimeError("GROQ_API_KEY environment variable is required")
    
    # Initialize Gmail agent
    try:
        from agents.gmail_agent import GmailAgent,Email
        gmail_agent = GmailAgent(
            credentials_file=config['gmail_credentials'],
            token_file=config['gmail_token']
        )
    except Exception as e:
        print(f"Warning: Gmail agent initialization failed: {e}")
        gmail_agent = None
    
    # Initialize Groq classifier
    try:
        from agents.filtering import GroqEmailClassifier
        classifier = GroqEmailClassifier(api_key=config['groq_api_key'])
    except Exception as e:
        print(f"Warning: Classifier initialization failed: {e}")
        classifier = None
    
    # Initialize knowledge graph if enabled
    if config['enable_knowledge_graph']:
        try:
            from agents.graph import EmailKnowledgeGraph
            knowledge_graph = EmailKnowledgeGraph(
                working_dir=config['rag_working_dir'],
                neo4j_uri=config['neo4j_uri'],
                neo4j_username=config['neo4j_username'],
                neo4j_password=config['neo4j_password'],
                llm_model=config['llm_model'],
                embedding_model=config['embedding_model'],
                ollama_host=config['ollama_host']
            )
            await knowledge_graph.initialize()
        except Exception as e:
            print(f"Warning: Knowledge graph initialization failed: {e}")
            knowledge_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await initialize_services()
    yield
    # Shutdown - cleanup if needed
    pass

# Create FastAPI app
app = FastAPI(
    title="Email Classification and Knowledge Graph API",
    description="API for classifying emails and processing important ones into a knowledge graph",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Email Classification and Knowledge Graph API",
        "version": "1.0.0",
        "knowledge_graph_enabled": config['enable_knowledge_graph'] if config else False,
        "services": {
            "gmail_agent": gmail_agent is not None,
            "classifier": classifier is not None,
            "knowledge_graph": knowledge_graph is not None
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "gmail_agent": gmail_agent is not None,
        "classifier": classifier is not None,
        "knowledge_graph": knowledge_graph is not None,
        "knowledge_graph_enabled": config['enable_knowledge_graph'] if config else False
    }
    return health_status

@app.post("/emails/fetch-and-classify")
async def fetch_and_classify_emails(request: EmailFetchRequest):
    """
    Fetch emails from Gmail and classify them
    """
    try:
        if not gmail_agent or not classifier:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # Fetch emails
        email_objects = gmail_agent.read_recent_emails(
            days=request.days_back,
            max_emails=request.max_emails
        )
        
        if not email_objects:
            return {"message": "No emails found", "emails": [], "important_emails": []}
        
        # Classify emails
        email_bodies = [email.body for email in email_objects]
        classifications = classifier.classify_batch(email_bodies)
        
        # Convert Email objects to dictionaries for JSON serialization
        email_dicts = []
        for email in email_objects:
            email_dict = {
                "subject": email.subject,
                "sender": email.sender,
                "body": email.body,
                "timestamp": email.timestamp
            }
            if hasattr(email, 'id'):
                email_dict["id"] = email.id
            email_dicts.append(email_dict)
        
        # Combine emails with classifications
        classified_emails = []
        important_emails = []
        
        for email_dict, classification in zip(email_dicts, classifications):
            email_with_classification = {
                "email": email_dict,
                "classification": classification
            }
            classified_emails.append(email_with_classification)
            
            # Check if email is important
            if (classification['classification'] == 'IMPORTANT' and 
                classification['confidence'] >= request.confidence_threshold):
                important_emails.append(email_with_classification)
        
        # Sort important emails by confidence
        important_emails.sort(key=lambda x: x['classification']['confidence'], reverse=True)
        
        return {
            "message": f"Successfully classified {len(email_objects)} emails",
            "total_emails": len(classified_emails),
            "important_emails_count": len(important_emails),
            "confidence_threshold": request.confidence_threshold,
            "emails": classified_emails,
            "important_emails": important_emails
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing emails: {str(e)}")

@app.post("/emails/process-important")
async def process_important_emails_endpoint(
    background_tasks: BackgroundTasks,
    request: EmailFetchRequest
):
    """
    Fetch, classify, and process important emails to knowledge graph
    """
    try:
        if not config['enable_knowledge_graph']:
            raise HTTPException(
                status_code=400, 
                detail="Knowledge graph is disabled. Set ENABLE_KNOWLEDGE_GRAPH=true"
            )
        
        if not gmail_agent or not classifier or not knowledge_graph:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # First classify emails to get important ones
        classification_result = await fetch_and_classify_emails(request)
        important_emails = classification_result["important_emails"]
        
        if not important_emails:
            return {
                "message": "No important emails found to process",
                "total_emails": classification_result["total_emails"],
                "important_emails_count": 0
            }
        
        # Convert to format expected by knowledge graph
        important_email_tuples = [
            (email_with_class["email"], email_with_class["classification"]) 
            for email_with_class in important_emails
        ]
        
        # Process in background
        background_tasks.add_task(
            process_to_knowledge_graph_background,
            important_email_tuples
        )
        
        return {
            "message": f"Started processing {len(important_emails)} important emails to knowledge graph",
            "total_emails": classification_result["total_emails"],
            "important_emails_count": len(important_emails),
            "status": "processing_started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing emails: {str(e)}")

async def process_to_knowledge_graph_background(important_email_tuples):
    """Background task to process emails to knowledge graph"""
    try:
        if knowledge_graph and gmail_agent:
            await knowledge_graph.process_important_emails(important_email_tuples, gmail_agent.service)
            print(f"✅ Successfully processed {len(important_email_tuples)} emails to knowledge graph")
    except Exception as e:
        print(f"❌ Error processing emails to knowledge graph: {e}")

@app.post("/knowledge-graph/query")
async def query_knowledge_graph(request: QueryRequest):
    """
    Query the knowledge graph
    """
    try:
        if not config['enable_knowledge_graph']:
            raise HTTPException(
                status_code=400, 
                detail="Knowledge graph is disabled. Set ENABLE_KNOWLEDGE_GRAPH=true"
            )
        
        if not knowledge_graph:
            raise HTTPException(status_code=500, detail="Knowledge graph not initialized")
        
        response = await knowledge_graph.query_knowledge_graph(request.query, mode=request.mode)
        
        return {
            "query": request.query,
            "mode": request.mode,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying knowledge graph: {str(e)}")

@app.get("/knowledge-graph/status")
async def knowledge_graph_status():
    """
    Get knowledge graph status
    """
    if not config or not config['enable_knowledge_graph']:
        return {
            "enabled": False,
            "message": "Knowledge graph is disabled"
        }
    
    return {
        "enabled": True,
        "initialized": knowledge_graph is not None,
        "working_dir": config['rag_working_dir'],
        "neo4j_uri": config['neo4j_uri'],
        "llm_model": config['llm_model'],
        "embedding_model": config['embedding_model']
    }

@app.get("/emails/recent/{days}")
async def get_recent_emails(days: int, max_emails: int = 20):
    """
    Get recent emails without classification
    """
    try:
        if not gmail_agent:
            raise HTTPException(status_code=500, detail="Gmail agent not initialized")
        
        email_objects = gmail_agent.read_recent_emails(days=days, max_emails=max_emails)
        
        # Convert to dictionaries for JSON serialization
        email_dicts = []
        for email in email_objects:
            email_dict = {
                "subject": email.subject,
                "sender": email.sender,
                "body": email.body,
                "timestamp": email.timestamp
            }
            if hasattr(email, 'id'):
                email_dict["id"] = email.id
            email_dicts.append(email_dict)
        
        return {
            "message": f"Fetched {len(email_objects)} emails from last {days} days",
            "count": len(email_objects),
            "emails": email_dicts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching emails: {str(e)}")

@app.post("/classify/text")
async def classify_text(request: TextClassificationRequest):
    """
    Classify arbitrary text using the email classifier
    """
    try:
        if not classifier:
            raise HTTPException(status_code=500, detail="Classifier not initialized")
        
        result = classifier.classify_email_text(request.text)
        
        return {
            "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "classification": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying text: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )