import { ErrorHandler } from './error-handler';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3002/api';

// Auth token functions with refresh token support
export function getAuthToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('access_token');
  }
  return null;
}

export function setAuthToken(token: string): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem('access_token', token);
  }
}

export function getRefreshToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('refresh_token');
  }
  return null;
}

export function setRefreshToken(token: string): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem('refresh_token', token);
  }
}

export function removeAuthTokens(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }
}

export function removeAuthToken(): void {
  removeAuthTokens();
}

// Enhanced API call wrapper with auth, timeout, and error handling
async function apiCall(endpoint: string, options: RequestInit = {}): Promise<Response> {
  const token = getAuthToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` }),
    ...options.headers,
  };

  const fullUrl = `${API_BASE_URL}${endpoint}`;
  console.log('API Call:', {
    url: fullUrl,
    method: options.method || 'GET',
    hasToken: !!token,
    headers: Object.keys(headers)
  });

  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minutes timeout

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    // Handle 401 Unauthorized - try to refresh token first
    if (response.status === 401) {
      const refreshed = await tryRefreshToken();
      if (refreshed) {
        // Retry the original request with new token
        const newToken = getAuthToken();
        return fetch(`${API_BASE_URL}${endpoint}`, {
          ...options,
          headers: {
            ...headers,
            'Authorization': `Bearer ${newToken}`
          },
          signal: controller.signal,
        });
      }
      
      // If refresh failed, remove tokens and redirect
      removeAuthTokens();
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
      throw new Error('Unauthorized');
    }

    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    
    // Enhanced error handling with ErrorHandler
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Request timeout - the server is taking too long to respond');
    }
    
    // Parse and handle error appropriately
    const parsedError = ErrorHandler.parseError(error);
    
    // For auth errors, automatically clear tokens
    if (parsedError.type === 'auth' && parsedError.action) {
      parsedError.action();
    }
    
    throw error;
  }
}

// Types
export interface User {
  user_id: number;
  username: string;
  emailaddress: string;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface Conversation {
  conversation_id: number;
  user_id: number;
  timestamp: string;
  messages: Message[];
  first_message?: string;
}

export interface LoginResponse {
  message: string;
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

export interface RegisterResponse {
  message: string;
  user: User;
}

// Auth API functions
export async function loginUser(username: string, password: string): Promise<LoginResponse> {
  const response = await fetch(`${API_BASE_URL}/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      username: username,
      password: password,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Login failed');
  }

  return response.json();
}

export async function registerUser(userData: {
  username: string;
  emailaddress: string;
  password: string;
}): Promise<RegisterResponse> {
  const response = await fetch(`${API_BASE_URL}/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userData),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Registration failed');
  }

  return response.json();
}

export async function getCurrentUser(): Promise<{ user: User }> {
  const response = await apiCall('/me');

  if (!response.ok) {
    throw new Error('Failed to get current user');
  }

  return response.json();
}

export async function logoutUser(): Promise<{ message: string }> {
  const response = await apiCall('/logout', {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error('Failed to logout');
  }

  removeAuthTokens();
  return response.json();
}

// Token refresh functionality
async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    return false;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        refresh_token: refreshToken,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      setAuthToken(data.access_token);
      return true;
    }
  } catch (error) {
    console.error('Token refresh failed:', error);
  }

  return false;
}

export async function refreshAccessToken(): Promise<{ access_token: string; token_type: string } | null> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    throw new Error('No refresh token available');
  }

  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      refresh_token: refreshToken,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Token refresh failed');
  }

  const data = await response.json();
  setAuthToken(data.access_token);
  return data;
}

export async function revokeRefreshToken(): Promise<{ message: string }> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    return { message: 'No refresh token to revoke' };
  }

  const response = await fetch(`${API_BASE_URL}/auth/revoke`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      refresh_token: refreshToken,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Token revocation failed');
  }

  return response.json();
}

// Conversation API functions
export async function getConversations(): Promise<Array<{
  conversation_id: number;
  user_id: number;
  timestamp: string;
  first_message: string;
}>> {
  const response = await apiCall('/conversations');

  if (!response.ok) {
    throw new Error('Failed to fetch conversations');
  }

  return response.json();
}

export async function createConversation(firstMessage: string): Promise<Conversation> {
  const response = await apiCall('/conversations', {
    method: 'POST',
    body: JSON.stringify({
      first_message: firstMessage,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create conversation');
  }

  return response.json();
}

export async function getConversation(conversationId: number): Promise<Conversation> {
  const response = await apiCall(`/conversations/${conversationId}`);

  if (!response.ok) {
    throw new Error('Failed to fetch conversation');
  }

  return response.json();
}

export async function deleteConversation(conversationId: number): Promise<{ message: string }> {
  const response = await apiCall(`/conversations/${conversationId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to delete conversation');
  }

  return response.json();
}

// Message API functions
export async function sendMessage(conversationId: number, content: string): Promise<ReadableStream<Uint8Array> | null> {
  const token = getAuthToken();
  if (!token) {
    throw new Error('No auth token available');
  }

  // Create abort controller for streaming timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minutes timeout

  try {
    const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        content: content,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to send message');
    }

    return response.body;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Message timeout - the AI is taking too long to respond');
    }
    throw error;
  }
}

export async function sendMessageNonStream(conversationId: number, content: string): Promise<Conversation> {
  const response = await apiCall(`/conversations/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify({
      content: content,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to send message');
  }

  return response.json();
}

// Chat streaming function with proper event parsing
export async function streamChat(
  conversationId: number, 
  message: string, 
  onChunk: (chunk: string) => void,
  onComplete?: (fullMessage: string) => void,
  onError?: (error: string) => void
): Promise<void> {
  try {
    const stream = await sendMessage(conversationId, message);
    if (!stream) {
      throw new Error('No stream received');
    }

    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let fullMessage = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          // Stream ended without completion signal - call completion handler
          console.log('Stream ended naturally, completing with full message');
          if (onComplete && fullMessage) {
            onComplete(fullMessage);
          }
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            if (data === '') continue;
            
            // Handle special case for [DONE] marker
            if (data === '[DONE]') {
              console.log('Stream completed with [DONE] marker');
              if (onComplete) {
                onComplete(fullMessage);
              }
              return;
            }
            
            try {
              const parsed = JSON.parse(data);
              console.log('Parsed streaming data:', parsed);
              
              switch (parsed.type) {
                case 'chunk':
                  if (parsed.content) {
                    fullMessage += parsed.content;
                    onChunk(parsed.content);
                  }
                  break;
                case 'assistant_chunk':
                  if (parsed.content) {
                    fullMessage += parsed.content;
                    onChunk(parsed.content);
                  }
                  break;
                case 'complete':
                  console.log('Complete signal received');
                  if (onComplete) {
                    onComplete(fullMessage);
                  }
                  return;
                case 'assistant_complete':
                  console.log('Assistant complete signal received');
                  if (onComplete) {
                    onComplete(fullMessage);
                  }
                  return;
                case 'done':
                  console.log('Done signal received');
                  if (onComplete) {
                    onComplete(fullMessage);
                  }
                  return;
                case 'end':
                  console.log('End signal received');
                  if (onComplete) {
                    onComplete(fullMessage);
                  }
                  return;
                case 'error':
                  if (onError) {
                    onError(parsed.error);
                  }
                  throw new Error(parsed.error);
                  break;
              }
            } catch (e) {
              // Ignore JSON parsing errors for incomplete chunks
              console.log('Failed to parse chunk:', data, 'Error:', e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error('Streaming error:', error);
    if (onError) {
      onError(error instanceof Error ? error.message : 'Unknown streaming error');
    }
    throw error;
  }
}
