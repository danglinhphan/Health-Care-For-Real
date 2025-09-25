import { create } from 'zustand';
import { getConversations, deleteConversation as apiDeleteConversation } from '@/lib/api';

export interface Conversation {
  conversation_id: number;
  user_id: number;
  timestamp: string;
  first_message: string;
}

export interface ConversationState {
  conversations: Conversation[];
  activeConversationId: number | null;
  isLoading: boolean;
  setConversations: (conversations: Conversation[]) => void;
  addConversation: (conversation: Conversation) => void;
  updateConversation: (id: number, updates: Partial<Conversation>) => void;
  deleteConversation: (id: number) => Promise<void>;
  setActiveConversation: (id: number | null) => void;
  setLoading: (loading: boolean) => void;
  refreshConversations: () => Promise<void>;
  getActiveConversation: () => Conversation | null;
}

export const useConversationStore = create<ConversationState>()((set, get) => ({
  conversations: [],
  activeConversationId: null,
  isLoading: false,
  
  setConversations: (conversations) => set({ conversations }),
  
  addConversation: (conversation) => set((state) => ({
    conversations: [conversation, ...state.conversations]
  })),
  
  updateConversation: (id, updates) => set((state) => ({
    conversations: state.conversations.map(conv =>
      conv.conversation_id === id ? { ...conv, ...updates } : conv
    )
  })),
  
  deleteConversation: async (id) => {
    try {
      await apiDeleteConversation(id);
      set((state) => ({
        conversations: state.conversations.filter(conv => conv.conversation_id !== id),
        activeConversationId: state.activeConversationId === id ? null : state.activeConversationId
      }));
    } catch (error) {
      console.error('Error deleting conversation:', error);
      throw error;
    }
  },
  
  setActiveConversation: (id) => set({ activeConversationId: id }),
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  refreshConversations: async () => {
    try {
      set({ isLoading: true });
      const conversations = await getConversations();
      // Ensure conversations is always an array
      set({ conversations: Array.isArray(conversations) ? conversations : [], isLoading: false });
    } catch (error) {
      console.error('Error refreshing conversations:', error);
      // Set empty array on error to prevent map errors
      set({ conversations: [], isLoading: false });
    }
  },
  
  getActiveConversation: () => {
    const state = get();
    return state.conversations.find(conv => conv.conversation_id === state.activeConversationId) || null;
  },
}));
