// chatbot.component.ts
import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

interface Message {
  text: string;
  sender: 'user' | 'bot';
}

interface Email {
  subject: string;
  sender: string;
  body: string;
  timestamp?: string;
}

@Component({
  selector: 'app-chat-window',  // <-- match template
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat-window.component.html',
  styleUrls: ['./chat-window.component.css']
})
export class ChatWindowComponent implements OnInit, AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;
  
  messages: Message[] = [];
  userMessage: string = '';
  isTyping: boolean = false;
  apiBaseUrl: string = 'http://localhost:8000';

  constructor(private http: HttpClient) {}
  
  ngOnInit() {
    // Add welcome message
    this.addBotMessage('Hello! I can help you analyze your important emails. Ask me questions about your processed emails.');
  }
  
  ngAfterViewChecked() {
    this.scrollToBottom();
  }
  
  scrollToBottom(): void {
    try {
      this.messagesContainer.nativeElement.scrollTop = this.messagesContainer.nativeElement.scrollHeight;
    } catch (err) {}
  }
  
  addMessage(text: string, sender: 'user' | 'bot') {
    this.messages.push({
      text,
      sender
    });
  }
  
  addBotMessage(text: string) {
    this.addMessage(text, 'bot');
  }
  
  addUserMessage(text: string) {
    this.addMessage(text, 'user');
  }
  
  async sendMessage() {
    if (!this.userMessage.trim() || this.isTyping) return;
    
    const message = this.userMessage.trim();
    this.addUserMessage(message);
    this.userMessage = '';
    
    this.isTyping = true;
    
    try {
      const response = await this.http.post<any>(
        `${this.apiBaseUrl}/ask`,
        { question: message }
      ).toPromise();
      
      this.addBotMessage(response.answer);
    } catch (error) {
      console.error('Error asking question:', error);
      this.addBotMessage('Sorry, I couldn\'t process your question. Make sure emails have been processed first.');
    } finally {
      this.isTyping = false;
    }
  }
}