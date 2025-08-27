# filtering_agent.py
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.ollama import Ollama
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Email(BaseModel):
    """Email data structure"""
    subject: str = Field(description="Email subject")
    sender: str = Field(description="Email sender")
    body: str = Field(description="Email body content")
    timestamp: Optional[str] = Field(default=None, description="Email timestamp")
    id: Optional[str] = Field(default=None, description="Gmail message ID")

class ClassificationResult(BaseModel):
    """Classification result structure"""
    classification: str = Field(description="IMPORTANT or NOT_IMPORTANT")
    topic: str = Field(description="WORK, PERSONAL, or PROMOTIONAL")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation for the classification")

class FilteringAgent:
    """Email classifier using Agno with Ollama"""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        """
        Initialize the Agno classifier with Ollama.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self.agent = self._create_agent(model, base_url)
        self.default_response = ClassificationResult(
            classification="NOT_IMPORTANT",
            topic="No topic provided",
            confidence=0.0,
            reasoning="Classification failed"
        )

    def _create_agent(self, model: str, base_url: str) -> Agent:
        """Create Agno agent for email classification"""
        return Agent(
            model=Ollama(model=model, base_url=base_url),
            response_model=ClassificationResult,
            description="""You are an expert email classifier. 

The emails belong to a computer science engineer working in an IT company. 

Topic definitions:
- WORK: Job-related communications, meetings, projects, company announcements
- PERSONAL: Personal correspondence, non-work related communications
- PROMOTIONAL: Marketing emails, newsletters, advertisements, sales pitches

Importance is based on relevance to their work and immediate action required.""",
            markdown=True
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _classify_email_with_agent(self, email: Email) -> ClassificationResult:
        """Classify email using Agno agent"""
        try:
            prompt = f"""
            Classify this email:
            Subject: {email.subject}
            From: {email.sender}
            Body: {email.body[:2000]}
            """
            
            result = self.agent.run(prompt)
            return result
        except Exception as e:
            logger.error(f"Agno classification failed: {e}")
            raise

    def classify_email(self, email: Email) -> Dict:
        """Classify a single email using Agno with Ollama"""
        try:
            result = self._classify_email_with_agent(email)
            
            return {
                "classification": result.classification,
                "topic": result.topic,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to classify email: {e}")
            return {
                "classification": self.default_response.classification,
                "topic": self.default_response.topic,
                "confidence": self.default_response.confidence,
                "reasoning": self.default_response.reasoning,
                "success": False
            }

    def classify_batch(self, emails: List[Email], batch_size: int = 5) -> List[Dict]:
        """
        Efficiently classify a list of emails in batches.

        Args:
            emails: List of Email objects.
            batch_size: Number of emails per batch.

        Returns:
            List of classification results.
        """
        if not emails:
            return []

        results = []
        total_emails = len(emails)

        for i in range(0, total_emails, batch_size):
            batch = emails[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_emails - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            for email in batch:
                try:
                    result = self.classify_email(email)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to classify email '{email.subject}': {e}")
                    results.append({
                        "classification": self.default_response.classification,
                        "topic": self.default_response.topic,
                        "confidence": self.default_response.confidence,
                        "reasoning": self.default_response.reasoning,
                        "success": False
                    })

        return results

    def get_important_emails(
        self,
        emails: List[Email],
        confidence_threshold: float = 0.7,
        max_results: Optional[int] = None
    ) -> List[Tuple[Email, Dict]]:
        """
        Filter and return important emails based on confidence threshold.

        Args:
            emails: List of Email objects.
            confidence_threshold: Minimum confidence score to consider important.
            max_results: Optional limit on number of results.

        Returns:
            List of (Email, classification result) tuples.
        """
        if not emails:
            return []

        if not (0 <= confidence_threshold <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1.")

        results = self.classify_batch(emails)
        important_emails = [
            (email, result)
            for email, result in zip(emails, results)
            if result["classification"] == "IMPORTANT" and result["confidence"] >= confidence_threshold
        ]

        # Sort by confidence descending
        important_emails.sort(key=lambda x: x[1]["confidence"], reverse=True)

        if max_results is not None:
            return important_emails[:max_results]
        return important_emails