import { create } from 'zustand';
import { ErrorHandler } from './error-handler';
import { useToast } from '@/hooks/use-toast';

// Generate unique ID for messages
function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
import { 
  sendMessage as apiSendMessage, 
  createConversation, 
  getConversation,
  streamChat,
  type Message as ApiMessage,
  type Conversation as ApiConversation
} from '@/lib/api';

export interface Message {
  id: string; // Changed to string for better uniqueness
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  streamingMessage: string;
  currentConversationId: number | null;
  error: string | null;
  
  // State management
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  setStreaming: (streaming: boolean) => void;
  setStreamingMessage: (message: string) => void;
  appendToStreamingMessage: (chunk: string) => void;
  clearStreamingMessage: () => void;
  clearMessages: () => void;
  clearChat: () => void;
  setCurrentConversation: (conversationId: number | null) => void;
  setError: (error: string | null) => void;
  
  // Actions
  loadConversation: (conversationId: number) => Promise<void>;
  sendMessage: (message: string, onConversationCreated?: (conversationId: number) => void) => Promise<void>;
}

export const useChatStore = create<ChatState>()((set, get) => ({
  messages: [],
  isLoading: false,
  isStreaming: false,
  streamingMessage: '',
  currentConversationId: null,
  error: null,
  
  setMessages: (messages) => set({ messages }),
  
  addMessage: (message) => set((state) => {
    // Simple check for exact ID duplicates only
    const isDuplicate = state.messages.some(existingMsg => existingMsg.id === message.id);
    
    if (isDuplicate) {
      console.log('Preventing duplicate message with same ID:', message.id);
      return state;
    }
    
    return {
      messages: [...state.messages, message]
    };
  }),
  
  setLoading: (isLoading) => set({ isLoading }),
  
  setStreaming: (isStreaming) => set({ isStreaming }),
  
  setStreamingMessage: (message) => set({ streamingMessage: message }),
  
  appendToStreamingMessage: (chunk) => set((state) => ({
    streamingMessage: state.streamingMessage + chunk
  })),
  
  clearStreamingMessage: () => set({ streamingMessage: '' }),
  
  clearMessages: () => set({ 
    messages: [], 
    streamingMessage: '',
    isLoading: false,
    isStreaming: false
  }),
  
  clearChat: () => {
    console.log('Chat store: Clearing chat and resetting conversation ID to null');
    set({ 
      messages: [], 
      streamingMessage: '', 
      currentConversationId: null,
      isLoading: false,
      isStreaming: false
    });
  },
  
  setCurrentConversation: (conversationId) => {
    console.log('Chat store: Setting current conversation to:', conversationId);
    set({ currentConversationId: conversationId });
  },
  
  setError: (error) => set({ error }),

  loadConversation: async (conversationId: number) => {
    set({ isLoading: true, isStreaming: false, streamingMessage: '', error: null });
    try {
      const userData = typeof window !== 'undefined' ? localStorage.getItem('user') : null;
      if (!userData) {
        throw new Error('User not authenticated');
      }
      
      const conversation = await getConversation(conversationId);
      
      // Convert API messages to store message format, ensure messages is an array
      const apiMessages = Array.isArray(conversation.messages) ? conversation.messages : [];
      const messages: Message[] = apiMessages.map((msg, index) => ({
        id: `${conversationId}_${index}_${msg.role}`, // Consistent ID based on conversation and position
        role: msg.role,
        content: msg.content,
        created_at: msg.timestamp,
      }));
      
      // Single state update to prevent multiple re-renders
      set({ 
        messages,
        currentConversationId: conversationId,
        isLoading: false,
        isStreaming: false,
        streamingMessage: '',
        error: null
      });
    } catch (error) {
      console.error('Error loading conversation:', error);
      const parsedError = ErrorHandler.parseError(error);
      const errorMessage = ErrorHandler.getErrorMessage(parsedError);
      
      set({ 
        isLoading: false,
        isStreaming: false,
        streamingMessage: '',
        error: errorMessage 
      });
      
      // Show user-friendly notification
      ErrorHandler.showErrorNotification(parsedError);
    }
  },

  sendMessage: async (message: string, onConversationCreated?: (conversationId: number) => void) => {
    const { currentConversationId } = get();
    console.log('Chat store: Starting sendMessage', { message, currentConversationId });
    set({ isLoading: true, isStreaming: true });

    try {
      const userData = typeof window !== 'undefined' ? localStorage.getItem('user') : null;
      console.log('Chat store: User data from localStorage:', userData);
      if (!userData) {
        throw new Error('User not authenticated');
      }

      const user = JSON.parse(userData);
      let conversationId = currentConversationId;

      // Create new conversation if none exists
      if (!conversationId) {
        const newConversation = await createConversation(message);
        conversationId = newConversation.conversation_id;
        set({ currentConversationId: conversationId });
        onConversationCreated?.(conversationId);
        
        // For new conversations, the backend already has both messages, ensure messages is an array
        const apiMessages = Array.isArray(newConversation.messages) ? newConversation.messages : [];
        const messages: Message[] = apiMessages.map((msg, index) => ({
          id: `${conversationId}_${index}_${msg.role}`, // Consistent ID based on conversation and position
          role: msg.role,
          content: msg.content,
          created_at: msg.timestamp,
        }));
        
        // Single state update to prevent multiple re-renders
        set({ 
          messages,
          currentConversationId: conversationId,
          isLoading: false,
          isStreaming: false,
          streamingMessage: '',
          error: null
        });
        return;
      }

      // For existing conversations, add user message and stream response
      const userMessage: Message = {
        id: generateMessageId(),
        role: 'user',
        content: message,
        created_at: new Date().toISOString(),
      };
      
      set((state) => ({
        messages: [...state.messages, userMessage]
      }));

      // Start streaming response
      set({ streamingMessage: '' });
      
      await streamChat(
        conversationId,
        message,
        // onChunk
        (chunk: string) => {
          set((state) => ({
            streamingMessage: state.streamingMessage + chunk
          }));
        },
        // onComplete
        (fullMessage: string) => {
          console.log('Chat store: onComplete called with message length:', fullMessage.length);
          const finalAssistantMessage: Message = {
            id: generateMessageId(), // Use consistent ID generation
            role: 'assistant',
            content: fullMessage,
            created_at: new Date().toISOString(),
          };

          // Use direct state update instead of function update to ensure immediate re-render
          const currentState = get();
          const newMessages = [...currentState.messages, finalAssistantMessage];
          
          console.log('Chat store: Adding assistant message, total messages will be:', newMessages.length);
          
          set({
            messages: newMessages,
            streamingMessage: '',
            isStreaming: false,
            isLoading: false,
            currentConversationId: conversationId
          });
        },
        // onError
        (error: string) => {
          console.error('Streaming error:', error);
          set({ isLoading: false, isStreaming: false });
          throw new Error(error);
        }
      );
    } catch (error) {
      console.error('Error sending message:', error);
      const parsedError = ErrorHandler.parseError(error);
      const errorMessage = ErrorHandler.getErrorMessage(parsedError);
      
      set({ 
        isLoading: false, 
        isStreaming: false,
        error: errorMessage 
      });
      
      // Show user-friendly notification
      ErrorHandler.showErrorNotification(parsedError);
      
      throw error;
    }
  }
}));
