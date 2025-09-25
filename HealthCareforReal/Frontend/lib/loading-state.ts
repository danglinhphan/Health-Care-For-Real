// Loading state machine for better UX
export type LoadingState = 
  | 'idle'
  | 'creating_conversation'
  | 'loading_conversation'
  | 'sending_message'
  | 'streaming_response'
  | 'error';

export interface LoadingStateContext {
  state: LoadingState;
  error?: string;
  isLoading: boolean;
  canSendMessage: boolean;
  loadingMessage: string;
}

export function getLoadingContext(state: LoadingState, error?: string): LoadingStateContext {
  switch (state) {
    case 'idle':
      return {
        state,
        isLoading: false,
        canSendMessage: true,
        loadingMessage: '',
      };
      
    case 'creating_conversation':
      return {
        state,
        isLoading: true,
        canSendMessage: false,
        loadingMessage: 'Starting new conversation...',
      };
      
    case 'loading_conversation':
      return {
        state,
        isLoading: true,
        canSendMessage: false,
        loadingMessage: 'Loading conversation history...',
      };
      
    case 'sending_message':
      return {
        state,
        isLoading: true,
        canSendMessage: false,
        loadingMessage: 'Sending message...',
      };
      
    case 'streaming_response':
      return {
        state,
        isLoading: false, // Allow new messages during streaming
        canSendMessage: true,
        loadingMessage: 'AI is responding...',
      };
      
    case 'error':
      return {
        state,
        error,
        isLoading: false,
        canSendMessage: true,
        loadingMessage: '',
      };
      
    default:
      return {
        state: 'idle',
        isLoading: false,
        canSendMessage: true,
        loadingMessage: '',
      };
  }
}