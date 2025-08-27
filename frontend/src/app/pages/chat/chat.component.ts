import { Component, ViewChild } from '@angular/core';
import { ChatWindowComponent } from '../../components/chat-window/chat-window.component';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';


interface Chat {
  id: number;
  title: string;
  editing?: boolean;
  messages?: { sender: 'user' | 'bot', text: string }[];
  lastActivity?: Date;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [ChatWindowComponent, CommonModule, FormsModule, HttpClientModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent {
  @ViewChild(ChatWindowComponent) chatWindow!: ChatWindowComponent;
   

  username = 'Ahmed';
  showUserMenu = false;
  activeChat: Chat | null = null;
  
  chatHistory: Chat[] = [
    { 
      id: 1, 
      title: 'Welcome Chat',
      lastActivity: new Date(),
      messages: [
        { sender: 'bot', text: 'Hello! How can I assist you today?' }
      ]
    }
  ];

  constructor() {
    this.activeChat = this.chatHistory[0];
  }

  toggleUserMenu() {
    this.showUserMenu = !this.showUserMenu;
  }

  toggleEdit(chat: Chat) {
    // Save any other chat that might be in edit mode
    this.chatHistory.forEach(c => {
      if (c.id !== chat.id) c.editing = false;
    });
    chat.editing = !chat.editing;
  }

  addNewChat() {
    const newChat: Chat = {
      id: Date.now(),
      title: `New Chat ${this.chatHistory.length + 1}`,
      editing: false,
      lastActivity: new Date(),
      messages: [
        { sender: 'bot', text: 'Hello! How can I assist you today?' }
      ]
    };
    
    this.chatHistory.unshift(newChat);
    this.loadChat(newChat);
  }

  deleteChat(id: number) {
    if (this.chatHistory.length <= 1) {
      alert('You must have at least one chat.');
      return;
    }

    const chatToDelete = this.chatHistory.find(chat => chat.id === id);
    if (chatToDelete && confirm(`Are you sure you want to delete "${chatToDelete.title}"?`)) {
      this.chatHistory = this.chatHistory.filter(chat => chat.id !== id);
      
      // If we deleted the active chat, switch to the first available chat
      if (this.activeChat?.id === id) {
        this.loadChat(this.chatHistory[0]);
      }
    }
  }

  saveChat(chat: Chat) {
    if (!chat.title.trim()) {
      chat.title = `Chat ${chat.id}`;
    }
    chat.editing = false;
  }

  loadChat(chat: Chat) {
    // Save current chat messages before switching
    if (this.activeChat && this.chatWindow) {
      this.activeChat.messages = [...this.chatWindow.messages];
      this.activeChat.lastActivity = new Date();
    }

    this.activeChat = chat;
    
    // Update chat window messages after view is initialized
    setTimeout(() => {
      if (this.chatWindow && chat.messages) {
        this.chatWindow.messages = [...chat.messages];
      }
    }, 0);
  }

  onKeydownChatTitle(event: KeyboardEvent, chat: Chat) {
    if (event.key === 'Enter') {
      this.saveChat(chat);
    } else if (event.key === 'Escape') {
      chat.editing = false;
    }
  }

  formatLastActivity(date: Date | undefined): string {
    if (!date) return '';
    
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    return `${Math.floor(diffInHours / 24)}d ago`;
  }
}