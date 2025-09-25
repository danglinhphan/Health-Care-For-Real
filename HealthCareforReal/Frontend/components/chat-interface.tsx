'use client';

import { useState, useRef, useEffect, useMemo, useLayoutEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Card } from '@/components/ui/card';
import { Send, Bot, User, Trash2, MessageSquare, Stethoscope, Heart, Activity } from 'lucide-react';
import { useChatStore } from '@/lib/chat-store';
import MessageContent from '@/components/message-content';
import { HealthcareDisclaimer, HealthcareDisclaimerBanner } from '@/components/healthcare-disclaimer';

interface ChatInterfaceProps {
  user: { user_id: number; username: string; emailaddress: string } | null;
  conversationId: number | null;
  onConversationCreated: (conversationId: number) => void;
}

export default function ChatInterface({ user, conversationId, onConversationCreated }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [lastMessageCount, setLastMessageCount] = useState(0);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const [showDisclaimer, setShowDisclaimer] = useState(true);
  const [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const prevConversationIdRef = useRef<number | null>(null);

  // Zustand store
  const {
    messages,
    streamingMessage,
    currentConversationId,
    setCurrentConversation,
    loadConversation,
    sendMessage,
    clearChat,
    isLoading,
    isStreaming
  } = useChatStore();

  // Check if disclaimer was previously accepted
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const accepted = localStorage.getItem('healthcare_disclaimer_accepted');
      if (accepted === 'true') {
        setDisclaimerAccepted(true);
        setShowDisclaimer(false);
      }
    }
  }, []);

  const handleDisclaimerAccept = () => {
    setDisclaimerAccepted(true);
    setShowDisclaimer(false);
    if (typeof window !== 'undefined') {
      localStorage.setItem('healthcare_disclaimer_accepted', 'true');
    }
  };

  const handleDisclaimerDecline = () => {
    // Redirect to a safe page or show alternative content
    window.location.href = '/';
  };

  // Load conversation when conversationId changes
  useEffect(() => {
    console.log('Effect triggered - conversationId:', conversationId);
    
    // Only run if conversationId actually changed
    if (prevConversationIdRef.current !== conversationId) {
      prevConversationIdRef.current = conversationId;
      
      if (conversationId) {
        console.log('Loading conversation:', conversationId);
        setIsTransitioning(true);
        setShouldAutoScroll(false); // Disable auto-scroll during load
        
        // Load conversation (loadConversation handles state clearing internally)
        loadConversation(conversationId).finally(() => {
          // Add delay to prevent flicker, then enable auto-scroll
          setTimeout(() => {
            setIsTransitioning(false);
            setShouldAutoScroll(true);
            // Scroll to bottom after everything is settled
            setTimeout(() => scrollToBottom(), 50);
          }, 150);
        });
      } else {
        console.log('Clearing conversation');
        setCurrentConversation(null);
        clearChat();
        setIsTransitioning(false);
        setShouldAutoScroll(true);
        setLastMessageCount(0);
      }
    }
  }, [conversationId, loadConversation, setCurrentConversation, clearChat]); // Add dependencies

  // Debug messages changes and force re-render on state changes
  useEffect(() => {
    console.log('ChatInterface messages updated:', (messages || []).length, 'messages');
    console.log('ChatInterface streaming state:', { isStreaming, isLoading, streamingMessage: streamingMessage?.length });
    // Debug first message content if it exists
    if (messages && messages.length > 0) {
      console.log('First message content preview:', messages[0].content.substring(0, 100) + '...');
      console.log('Last message content preview:', messages[messages.length - 1].content.substring(0, 100) + '...');
    }
    
    // Force component to re-render by updating a local state if needed
    setLastMessageCount((messages || []).length);
  }, [messages, isStreaming, isLoading, streamingMessage]);

  // Function to scroll to bottom
  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTo({
          top: scrollElement.scrollHeight,
          behavior: 'auto'
        });
      }
    }
  };

  // Smart auto-scroll: only when new messages are added or streaming
  useEffect(() => {
    if (!shouldAutoScroll || isTransitioning) return;
    
    const currentLength = (messages || []).length;
    // Check if messages count increased (new message added)
    if (currentLength > lastMessageCount) {
      setLastMessageCount(currentLength);
      // Small delay to ensure DOM is updated
      requestAnimationFrame(() => scrollToBottom());
    }
  }, [(messages || []).length, lastMessageCount, shouldAutoScroll, isTransitioning]);

  // Scroll during streaming - throttled to prevent excessive calls
  useEffect(() => {
    if (isStreaming && shouldAutoScroll && !isTransitioning) {
      const timeoutId = setTimeout(() => scrollToBottom(), 50);
      return () => clearTimeout(timeoutId);
    }
  }, [isStreaming, shouldAutoScroll, isTransitioning, streamingMessage]);

  // Keyboard navigation handler
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading) {
        handleSubmit(e as any);
      }
    }
    if (e.key === 'Escape') {
      setInput('');
      inputRef.current?.blur();
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const messageContent = input.trim();
    setInput('');

    // Enable auto-scroll for new message
    setShouldAutoScroll(true);

    // Debug logging
    console.log('Submitting message:', messageContent);
    console.log('Current conversation ID:', conversationId);
    console.log('User:', user);

    try {
      await sendMessage(messageContent, (newConversationId) => {
        console.log('New conversation created:', newConversationId);
        onConversationCreated(newConversationId);
      });
      console.log('Message sent successfully');
    } catch (error) {
      console.error('Error sending message:', error);
      console.error('Error details:', error instanceof Error ? error.message : 'Unknown error');
      // Re-enable input on error
      setInput(messageContent);
    }
  };

  const formatTime = (timestamp: string) => {
    const messageDate = new Date(timestamp);
    const now = new Date();
    const diffInMs = now.getTime() - messageDate.getTime();
    const diffInMinutes = Math.floor(diffInMs / (1000 * 60));
    const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60));
    const diffInDays = Math.floor(diffInMs / (1000 * 60 * 60 * 24));

    // If message is from today
    if (diffInDays === 0) {
      if (diffInMinutes < 1) {
        return 'Just now';
      } else if (diffInMinutes < 60) {
        return `${diffInMinutes}m ago`;
      } else {
        return messageDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      }
    }
    // If message is from yesterday
    else if (diffInDays === 1) {
      return `Yesterday ${messageDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    // If message is older
    else {
      return messageDate.toLocaleDateString([], { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit', 
        minute: '2-digit' 
      });
    }
  };

  // Memoize messages rendering to prevent unnecessary re-renders
  const memoizedMessages = useMemo(() => {
    console.log('Memoizing messages:', (messages || []).length, 'messages', 'conversationId:', conversationId);
    
    if (!messages || messages.length === 0) {
      return [];
    }
    
    return messages.map((message, index) => (
      <div
        key={message.id}
        className={`flex items-start space-x-4 transition-all duration-200 ${
          message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
        }`}
        role="article"
        aria-label={`${message.role === 'user' ? 'User' : 'AI Assistant'} message`}
      >
        {/* Avatar */}
        <Avatar 
          className={`w-10 h-10 flex-shrink-0 ${
            message.role === 'user' ? 'bg-blue-600' : 'bg-gray-600'
          }`}
          aria-hidden="true"
        >
          <AvatarFallback>
            {message.role === 'user' ? (
              <User className="w-5 h-5 text-white" />
            ) : (
              <Stethoscope className="w-5 h-5 text-white" />
            )}
          </AvatarFallback>
        </Avatar>
        
        {/* Message Content */}
        <div className={`flex-1 min-w-0 ${
          message.role === 'user' ? 'text-right' : ''
        }`}>
          <div className={`inline-block max-w-[85%] ${
            message.role === 'user' ? 'ml-auto' : 'mr-auto'
          }`}>
            <div
              className={`p-4 rounded-2xl ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-gray-50 border border-gray-100 shadow-sm'
              }`}
            >
              <MessageContent 
                content={message.content} 
                isUser={message.role === 'user'} 
              />
              <p
                className={`text-xs mt-3 opacity-70 ${
                  message.role === 'user'
                    ? 'text-blue-100'
                    : 'text-gray-500'
                }`}
              >
                {formatTime(message.created_at)}
              </p>
            </div>
          </div>
        </div>
      </div>
    ));
  }, [messages, isTransitioning, conversationId, isStreaming, streamingMessage]); // Re-memoize when messages, transition state, streaming state, or conversation changes

  return (
    <>
      <HealthcareDisclaimer 
        isOpen={showDisclaimer && !disclaimerAccepted}
        onAccept={handleDisclaimerAccept}
        onDecline={handleDisclaimerDecline}
      />
      
      <div className="flex flex-col h-[calc(100vh-4rem)] max-h-[900px] w-full max-w-5xl mx-auto bg-white border rounded-lg shadow-lg">
        {/* Healthcare Disclaimer Banner */}
        {disclaimerAccepted && <HealthcareDisclaimerBanner />}
        
        {/* Chat Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gray-50 rounded-t-lg">
        <div className="flex items-center space-x-3">
          {/* Debug: Quick Login for Testing */}
          {process.env.NODE_ENV === 'development' && !user && (
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                try {
                  const response = await fetch('http://localhost:3002/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: 'testuser', password: 'testpass123' })
                  });
                  const data = await response.json();
                  localStorage.setItem('access_token', data.access_token);
                  localStorage.setItem('user', JSON.stringify(data.user));
                  window.location.reload();
                } catch (error) {
                  console.error('Quick login failed:', error);
                }
              }}
              className="bg-green-100 hover:bg-green-200 text-green-800"
            >
              🔧 Quick Login (Dev)
            </Button>
          )}
        </div>
        {(messages || []).length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={clearChat}
            className="flex items-center space-x-2 text-red-600 hover:text-red-700 hover:bg-red-50"
          >
            <Trash2 className="w-4 h-4" />
            <span>Clear Chat</span>
          </Button>
        )}
      </div>

      {/* Messages Area - Fixed Height with Scroll */}
      <ScrollArea 
        className="flex-1 p-6 h-0 min-h-[500px]" 
        ref={scrollAreaRef} 
        key={currentConversationId}
        role="log"
        aria-live="polite"
        aria-label="Chat conversation messages"
      >
        {isLoading && (messages || []).length === 0 && conversationId ? (
          <div className="flex items-center justify-center h-full" role="status" aria-live="polite">
            <div className="flex flex-col items-center space-y-4">
              <div 
                className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"
                aria-hidden="true"
              ></div>
              <p className="text-gray-500">Loading conversation...</p>
            </div>
          </div>
        ) : (messages || []).length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center mb-6 shadow-lg">
              <Stethoscope className="w-10 h-10 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">
              Welcome to Healthcare AI Assistant!
            </h3>
            <p className="text-gray-600 max-w-md mb-6">
              Start a conversation with our healthcare AI assistant. Ask medical questions, get help with diagnosis, or learn about health topics!
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-md w-full">
              <button 
                onClick={() => setInput("Hello! How can you help me with healthcare questions?")}
                className="p-3 bg-blue-50 hover:bg-blue-100 rounded-lg text-blue-700 text-sm transition-colors"
              >
                🩺 Say hello
              </button>
              <button 
                onClick={() => setInput("Can you help me understand medical symptoms?")}
                className="p-3 bg-green-50 hover:bg-green-100 rounded-lg text-green-700 text-sm transition-colors"
              >
                💊 Medical help
              </button>
              <button 
                onClick={() => setInput("Explain a medical condition to me")}
                className="p-3 bg-purple-50 hover:bg-purple-100 rounded-lg text-purple-700 text-sm transition-colors"
              >
                🧠 Health education
              </button>
              <button 
                onClick={() => setInput("Help me with a healthcare question")}
                className="p-3 bg-red-50 hover:bg-red-100 rounded-lg text-red-700 text-sm transition-colors"
              >
                ❤️ Ask questions
              </button>
            </div>
          </div>
        ) : (
          <div className={`space-y-6 transition-opacity duration-200 ${isTransitioning ? 'opacity-50' : 'opacity-100'}`}>
            {memoizedMessages}
            
            {(isLoading || isStreaming) && (
              <div key={`streaming-${isStreaming}-${streamingMessage?.length || 0}`} className="flex items-start space-x-4">
                <Avatar className="w-10 h-10 bg-gray-600 flex-shrink-0">
                  <AvatarFallback>
                    <Stethoscope className="w-5 h-5 text-white" />
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <div className="inline-block max-w-[85%]">
                    <div className="p-4 rounded-2xl bg-gray-50 border border-gray-100 shadow-sm">
                      {streamingMessage ? (
                        <>
                          <MessageContent 
                            content={streamingMessage} 
                            isUser={false} 
                            key={`streaming-content-${streamingMessage.length}`}
                          />
                          <div className="flex items-center mt-3 text-xs text-gray-500 opacity-70">
                            <div className="flex space-x-1 mr-2">
                              <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse"></div>
                              <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.15s' }}></div>
                              <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.3s' }}></div>
                            </div>
                            <span>AI is responding...</span>
                          </div>
                        </>
                      ) : (
                        <div className="flex items-center py-2">
                          <div className="flex space-x-1.5 mr-3">
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                          </div>
                          <span className="text-sm text-gray-600">AI is thinking...</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 border-t bg-gray-50 rounded-b-lg">
        <form 
          onSubmit={handleSubmit} 
          className="flex space-x-3"
          aria-label="Send message form"
        >
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? "You can continue typing..." : "Type your message here..."}
            disabled={isLoading}
            className="flex-1 bg-white border-gray-300 focus:border-blue-500 focus:ring-blue-500"
            maxLength={1000}
            aria-label="Type your message"
            aria-describedby="message-help"
          />
          <Button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-6 bg-blue-600 hover:bg-blue-700 focus:ring-blue-500"
            aria-label={isLoading ? "Sending message" : "Send message"}
          >
            {isLoading ? (
              <div 
                className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"
                aria-hidden="true"
              />
            ) : (
              <Send className="w-4 h-4" aria-hidden="true" />
            )}
          </Button>
        </form>
        <div className="flex justify-between items-center mt-2">
          <p 
            className="text-xs text-gray-500"
            id="message-help"
            aria-live="polite"
          >
            {isStreaming ? (
              <span className="text-blue-600 flex items-center">
                <span 
                  className="w-2 h-2 bg-blue-600 rounded-full animate-pulse mr-2 block"
                  aria-hidden="true"
                ></span>
                AI is responding... You can continue the conversation
              </span>
            ) : (
              <>Press Enter to send</>
            )}
          </p>
          <p 
            className="text-xs text-gray-400"
            aria-label={`Character count: ${input.length} of 1000`}
          >
            {input.length}/1000
          </p>
        </div>
      </div>
      </div>
    </>
  );
}
