'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card } from '@/components/ui/card';
import { Plus, MessageSquare, Trash2, Menu } from 'lucide-react';
import { useConversationStore } from '@/lib/conversation-store';

interface ConversationSidebarProps {
  currentConversationId: number | null;
  onSelectConversation: (conversationId: number | null) => void;
  onNewConversation: () => void;
  refreshTrigger?: number;
}

export default function ConversationSidebar({ 
  currentConversationId, 
  onSelectConversation, 
  onNewConversation,
  refreshTrigger
}: ConversationSidebarProps) {
  
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  // Zustand store
  const {
    conversations,
    isLoading,
    deleteConversation,
    refreshConversations
  } = useConversationStore();

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  // Refresh when refreshTrigger changes - but only if not currently streaming
  useEffect(() => {
    if (refreshTrigger) {
      // Small delay to ensure streaming is complete
      const timeoutId = setTimeout(() => {
        refreshConversations();
      }, 500);
      return () => clearTimeout(timeoutId);
    }
  }, [refreshTrigger, refreshConversations]);

  const handleDeleteConversation = async (conversationId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteConversation(conversationId);
      if (currentConversationId === conversationId) {
        onSelectConversation(null);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const handleNewConversation = () => {
    onNewConversation();
  };

  const handleToggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString();
  };

  const truncateMessage = (message: string, maxLength: number = 50) => {
    return message.length > maxLength ? message.substring(0, maxLength) + '...' : message;
  };

  if (isCollapsed) {
    return (
      <div className="w-16 bg-gray-50 border-r border-gray-200 flex flex-col items-center p-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleToggleCollapse}
          className="w-full h-10 mb-2"
        >
          <Menu className="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleNewConversation}
          className="w-full h-10"
        >
          <Plus className="w-4 h-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="w-80 bg-gray-50 border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-gray-900">Chat Sessions</h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleToggleCollapse}
          >
            <Menu className="w-4 h-4" />
          </Button>
        </div>
        <Button
          onClick={handleNewConversation}
          className="w-full flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </Button>
      </div>

      {/* Conversations List */}
      <ScrollArea className="flex-1 p-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          </div>
        ) : !conversations || conversations.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No conversations yet</p>
            <p className="text-xs">Start a new chat to get started</p>
          </div>
        ) : (
          <div className="space-y-2">
            {(conversations || []).map((conversation) => (
              <Card
                key={conversation.conversation_id}
                className={`p-3 cursor-pointer transition-colors hover:bg-gray-100 group ${
                  currentConversationId === conversation.conversation_id
                    ? 'bg-blue-50 border-blue-200'
                    : 'bg-white'
                }`}
                onClick={() => onSelectConversation(conversation.conversation_id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <MessageSquare className="w-4 h-4 text-gray-400 flex-shrink-0" />
                      <span className="text-xs text-gray-500">
                        {formatTime(conversation.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-900 break-words">
                      {truncateMessage(conversation.first_message)}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => handleDeleteConversation(conversation.conversation_id, e)}
                    className="ml-2 p-1 opacity-0 group-hover:opacity-100 hover:bg-red-100 hover:text-red-600"
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
