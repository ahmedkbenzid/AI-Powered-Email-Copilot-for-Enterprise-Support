# app.py - FastAPI main application
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from agents.gmail_agent import GmailAgent
from agents.filtering import FilteringAgent
from agents.graph import EmailKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    success: bool

class EmailSyncRequest(BaseModel):
    days: int = 7
    max_emails: int = 20
    user_id: Optional[str] = "default_user"

class EmailSyncResponse(BaseModel):
    total_emails: int
    important_emails: int
    sync_timestamp: str
    success: bool
    message: str

class ImportantEmail(BaseModel):
    id: Optional[str]
    subject: str
    sender: str
    timestamp: Optional[str]
    topic: str
    confidence: float
    reasoning: str
    has_attachments: bool
    body_preview: str

class EmailListResponse(BaseModel):
    emails: List[ImportantEmail]
    total_count: int
    success: bool

# Global variables for agents (will be initialized on startup)
gmail_agent: Optional[GmailAgent] = None
filtering_agent: Optional[FilteringAgent] = None
knowledge_graph: Optional[EmailKnowledgeGraph] = None

# Background task status tracking
sync_status = {
    "is_running": False,
    "last_sync": None,
    "last_error": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup"""
    global gmail_agent, filtering_agent, knowledge_graph
    
    try:
        logger.info("Initializing email processing agents...")
        
        # Initialize Gmail agent
        gmail_agent = GmailAgent(
            credentials_file='credentials.json',
            token_file='token.pickle'
        )
        
        # Initialize filtering agent
        filtering_agent = FilteringAgent(
            model="llama3.1",
            base_url="http://localhost:11434"
        )
        
        # Initialize knowledge graph
        knowledge_graph = EmailKnowledgeGraph(
            working_dir="./rag_storage_neo4j",
            neo4j_uri="neo4j://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="graph2025",
            llm_model="llama3.2:3b",
            embedding_model="nomic-embed-text",
            ollama_host="http://localhost:11434"
        )
        await knowledge_graph.initialize()
        
        logger.info("All agents initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Email Chatbot API",
    description="FastAPI backend for email processing and chatbot interactions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (optional - for authentication)
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional authentication dependency"""
    # Implement your authentication logic here
    return {"user_id": "authenticated_user"}

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_initialized": all([gmail_agent, filtering_agent, knowledge_graph])
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_emails(message: ChatMessage):
    """Main chat endpoint for querying email knowledge"""
    try:
        if not knowledge_graph:
            raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
        
        # Query the knowledge graph
        response = await knowledge_graph.query_knowledge_graph(
            query=message.message,
            mode="hybrid"
        )
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync-emails", response_model=EmailSyncResponse)
async def trigger_email_sync(
    request: EmailSyncRequest,
    background_tasks: BackgroundTasks
):
    """Trigger email synchronization in background"""
    global sync_status
    
    if sync_status["is_running"]:
        raise HTTPException(
            status_code=409, 
            detail="Email sync is already running"
        )
    
    # Add background task
    background_tasks.add_task(
        sync_emails_background,
        request.days,
        request.max_emails,
        request.user_id
    )
    
    sync_status["is_running"] = True
    
    return EmailSyncResponse(
        total_emails=0,
        important_emails=0,
        sync_timestamp=datetime.now().isoformat(),
        success=True,
        message="Email sync started in background"
    )

@app.get("/sync-status")
async def get_sync_status():
    """Get current sync status"""
    return {
        "is_running": sync_status["is_running"],
        "last_sync": sync_status["last_sync"],
        "last_error": sync_status["last_error"]
    }

@app.get("/important-emails", response_model=EmailListResponse)
async def get_important_emails(
    limit: int = 10,
    offset: int = 0
):
    """Get list of important emails"""
    try:
        if not gmail_agent or not filtering_agent:
            raise HTTPException(status_code=503, detail="Agents not initialized")
        
        # This would typically come from a database
        # For now, we'll do a quick fetch
        emails = gmail_agent.read_recent_emails(days=7, max_emails=limit + offset)
        important = filtering_agent.get_important_emails(emails, confidence_threshold=0.7)
        
        # Convert to response format
        email_list = []
        for email, classification in important[offset:offset+limit]:
            has_attachments = False
            if email.id:
                has_attachments = knowledge_graph.check_email_attachments(
                    gmail_agent.service, email.id
                )
            
            email_list.append(ImportantEmail(
                id=email.id,
                subject=email.subject,
                sender=email.sender,
                timestamp=email.timestamp,
                topic=classification["topic"],
                confidence=classification["confidence"],
                reasoning=classification["reasoning"],
                has_attachments=has_attachments,
                body_preview=email.body[:200] + "..." if len(email.body) > 200 else email.body
            ))
        
        return EmailListResponse(
            emails=email_list,
            total_count=len(important),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error fetching important emails: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/email/{email_id}")
async def get_email_details(email_id: str):
    """Get full email details by ID"""
    try:
        if not gmail_agent:
            raise HTTPException(status_code=503, detail="Gmail agent not initialized")
        
        email = gmail_agent.get_message_details(email_id)
        if not email:
            raise HTTPException(status_code=404, detail="Email not found")
        
        return {
            "email": email.dict(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error fetching email {email_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def sync_emails_background(days: int, max_emails: int, user_id: str):
    """Background task for email synchronization"""
    global sync_status
    
    try:
        logger.info(f"Starting email sync for user {user_id}")
        
        # Read emails
        emails = gmail_agent.read_recent_emails(days=days, max_emails=max_emails)
        
        # Filter important emails
        important_emails = filtering_agent.get_important_emails(
            emails,
            confidence_threshold=0.7,
            max_results=10
        )
        
        # Process in knowledge graph
        await knowledge_graph.process_important_emails(
            important_emails,
            gmail_agent.service
        )
        
        # Update status
        sync_status.update({
            "is_running": False,
            "last_sync": datetime.now().isoformat(),
            "last_error": None
        })
        
        logger.info(f"Email sync completed. Processed {len(important_emails)} important emails")
        
    except Exception as e:
        logger.error(f"Email sync failed: {e}")
        sync_status.update({
            "is_running": False,
            "last_error": str(e)
        })


# WebSocket endpoint for real-time chat (optional)
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            if knowledge_graph:
                response = await knowledge_graph.query_knowledge_graph(
                    query=message_data.get("message", ""),
                    mode="hybrid"
                )
                
                await websocket.send_text(json.dumps({
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }))
            else:
                await websocket.send_text(json.dumps({
                    "response": "Knowledge graph not available",
                    "timestamp": datetime.now().isoformat(),
                    "success": False
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )