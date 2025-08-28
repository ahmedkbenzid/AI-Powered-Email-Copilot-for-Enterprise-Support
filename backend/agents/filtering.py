import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqEmailClassifier:
    """Email classifier using Groq's ultra-fast LLM inference."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Groq classifier.

        Args:
            api_key: Groq API key.
            model: Groq-supported model name.
        """
        if not api_key:
            raise ValueError("Groq API key is required.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.system_prompt = self._create_system_prompt()
        self.default_response = {
            "classification": "NOT_IMPORTANT",
            "topic": "No topic provided",
            "confidence": 0.0,
            "reasoning": "Classification failed",
            "success": False
        }

    def _create_system_prompt(self) -> str:
        """System prompt for Groq."""
        return """You are an expert email classifier. Respond STRICTLY in JSON format:
{
    "classification": "IMPORTANT" or "NOT_IMPORTANT",
    "topic": "WORK" or "PERSONAL" or "PROMOTIONAL",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}

The emails belong to a computer science engineer working in an IT company.

Topic definitions:
- WORK: Job-related communications, meetings, projects, company announcements
- PERSONAL: Personal correspondence, non-work related communications
- PROMOTIONAL: Marketing emails, newsletters, advertisements, sales pitches

Importance is based on relevance to their work and immediate action required.
"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_groq_api(self, user_prompt: str) -> Dict:
        """Call Groq API with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150,
            )
            parsed = json.loads(response.choices[0].message.content)
            return parsed
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

    def classify_email_text(self, email_text: str) -> Dict:
        """Classify email text using Groq."""
        try:
            user_prompt = f"""
            Email to classify:
            {email_text.strip()}
            """

            result = self._call_groq_api(user_prompt)

            # Validate keys
            classification = result.get("classification", "NOT_IMPORTANT").upper()
            if classification not in ["IMPORTANT", "NOT_IMPORTANT"]:
                raise ValueError("Invalid classification returned from model.")

            return {
                "classification": classification,
                "topic": result.get("topic", "No topic provided"),
                "confidence": float(result.get("confidence", 0.0)),
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to classify email: {e}")
            return self.default_response.copy()

    def classify_batch(self, email_texts: List[str], batch_size: int = 5) -> List[Dict]:
        """
        Efficiently classify a list of email texts in batches.

        Args:
            email_texts: List of email text strings.
            batch_size: Number of emails per batch.

        Returns:
            List of classification results.
        """
        if not email_texts:
            return []

        results = []
        total_emails = len(email_texts)

        for i in range(0, total_emails, batch_size):
            batch = email_texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_emails - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            for email_text in batch:
                try:
                    result = self.classify_email_text(email_text)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to classify email: {e}")
                    results.append(self.default_response.copy())

        return results

    def get_important_emails(
        self,
        email_texts: List[str],
        confidence_threshold: float = 0.7,
        max_results: Optional[int] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Filter and return important emails based on confidence threshold.

        Args:
            email_texts: List of email text strings.
            confidence_threshold: Minimum confidence score to consider important.
            max_results: Optional limit on number of results.

        Returns:
            List of (email_text, classification result) tuples.
        """
        if not email_texts:
            return []

        if not (0 <= confidence_threshold <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1.")

        results = self.classify_batch(email_texts)
        important_emails = [
            (email_text, result)
            for email_text, result in zip(email_texts, results)
            if result["classification"] == "IMPORTANT" and result["confidence"] >= confidence_threshold
        ]

        # Sort by confidence descending
        important_emails.sort(key=lambda x: x[1]["confidence"], reverse=True)

        if max_results is not None:
            return important_emails[:max_results]
        return important_emails