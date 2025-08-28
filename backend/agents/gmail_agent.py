# gmail_agent.py
import os
import base64
import pickle
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from agno.agent import Agent


class Email(BaseModel):
    """Email data structure using Pydantic"""
    id: Optional[str] = Field(default=None, description="Gmail message ID")
    subject: str = Field(description="Email subject")
    sender: str = Field(description="Email sender")
    body: str = Field(description="Email body content")
    timestamp: Optional[str] = Field(default=None, description="Email timestamp")

class GmailAgent:
    """
    AI Agent for connecting to Gmail and reading emails
    """
    
    # Gmail API scopes
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        """
        Initialize the Gmail AI Agent
        
        Args:
            credentials_file: Path to your Google API credentials JSON file
            token_file: Path to store the authentication token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """
        Authenticate with Gmail API using OAuth2
        """
        creds = None
        
        # Load existing token if available
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, request authorization
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(f"Credentials file '{self.credentials_file}' not found. "
                                          "Please download it from Google Cloud Console.")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build the Gmail service
        try:
            self.service = build('gmail', 'v1', credentials=creds)
            print("✅ Successfully authenticated with Gmail API")
        except HttpError as error:
            print(f"❌ Error building Gmail service: {error}")
            raise
    
    def get_messages(self, query='', max_results=10):
        """
        Get list of messages based on query
        
        Args:
            query: Gmail search query (e.g., 'is:unread', 'from:example@gmail.com')
            max_results: Maximum number of messages to retrieve
            
        Returns:
            List of message IDs and thread IDs
        """
        try:
            results = self.service.users().messages().list(
                userId='me', 
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            print(f"📧 Found {len(messages)} messages")
            return messages
            
        except HttpError as error:
            print(f"❌ Error retrieving messages: {error}")
            return []
    
    def get_message_details(self, message_id):
        """
        Get detailed information about a specific message
        
        Args:
            message_id: Gmail message ID
            
        Returns:
            Email object containing message details
        """
        try:
            message = self.service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            
            # Extract message details
            payload = message['payload']
            headers = payload.get('headers', [])
            
            # Get common headers
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
            
            # Get message body
            body = self.extract_message_body(payload)
            
            return Email(
                id=message_id,
                subject=subject,
                sender=sender,
                body=body,
                timestamp=date
            )
            
        except HttpError as error:
            print(f"❌ Error retrieving message {message_id}: {error}")
            return None
    
    def extract_message_body(self, payload):
        """
        Extract the body text from message payload
        
        Args:
            payload: Gmail message payload
            
        Returns:
            Decoded message body text
        """
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
                elif part['mimeType'] == 'text/html' and not body:
                    if 'data' in part['body']:
                        # Extract text from HTML (simple version)
                        html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        # Simple HTML to text conversion
                        import re
                        body = re.sub('<[^<]+?>', '', html_content)
                        break
        else:
            if payload['body'].get('data'):
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body
    
    def read_recent_emails(self, days=7, max_emails=20):
        """
        Read recent emails from the specified number of days
        
        Args:
            days: Number of days to look back
            max_emails: Maximum number of emails to process
            
        Returns:
            List of Email objects
        """
        # Calculate date range
        start_date = datetime.now() - timedelta(days=days)
        date_query = start_date.strftime('%Y/%m/%d')
        
        # Build query for recent emails
        query = f'after:{date_query}'
        
        print(f"🔍 Searching for emails after {date_query}")
        
        # Get message list
        messages = self.get_messages(query=query, max_results=max_emails)
        
        # Get detailed information for each message
        email_objects = []
        for msg in messages:
            email = self.get_message_details(msg['id'])
            if email:
                agent = Agent(
                    email=email
                )
                email_objects.append(email)
        
        return email_objects