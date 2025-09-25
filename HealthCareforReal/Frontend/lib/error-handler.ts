// Enhanced error handling for the healthcare chatbot application

export interface AppError {
  type: 'network' | 'auth' | 'validation' | 'server' | 'client';
  message: string;
  code?: string | number;
  recoverable: boolean;
  action?: () => void;
  details?: any;
}

export class ErrorHandler {
  /**
   * Parse and categorize different types of errors
   */
  static parseError(error: unknown): AppError {
    // Handle different error types
    if (error instanceof Response) {
      return this.handleHttpError(error);
    }
    
    if (error instanceof Error) {
      return this.handleJavaScriptError(error);
    }
    
    if (typeof error === 'string') {
      return {
        type: 'client',
        message: error,
        recoverable: true
      };
    }
    
    // Unknown error type
    return {
      type: 'client',
      message: 'An unexpected error occurred',
      recoverable: true,
      details: error
    };
  }

  /**
   * Handle HTTP response errors
   */
  private static handleHttpError(response: Response): AppError {
    const status = response.status;
    
    switch (status) {
      case 401:
        return {
          type: 'auth',
          message: 'Your session has expired. Please log in again.',
          code: status,
          recoverable: true,
          action: () => {
            // Clear stored auth data
            if (typeof window !== 'undefined') {
              localStorage.removeItem('access_token');
              localStorage.removeItem('user');
              window.location.href = '/login';
            }
          }
        };
        
      case 403:
        return {
          type: 'auth',
          message: 'You do not have permission to perform this action.',
          code: status,
          recoverable: false
        };
        
      case 404:
        return {
          type: 'client',
          message: 'The requested resource was not found.',
          code: status,
          recoverable: true
        };
        
      case 429:
        return {
          type: 'server',
          message: 'Too many requests. Please wait a moment and try again.',
          code: status,
          recoverable: true,
          action: () => {
            // Could implement automatic retry with backoff
            setTimeout(() => window.location.reload(), 5000);
          }
        };
        
      case 500:
      case 502:
      case 503:
      case 504:
        return {
          type: 'server',
          message: 'Server error. Our team has been notified. Please try again later.',
          code: status,
          recoverable: true
        };
        
      default:
        return {
          type: 'network',
          message: `Request failed with status ${status}`,
          code: status,
          recoverable: true
        };
    }
  }

  /**
   * Handle JavaScript errors
   */
  private static handleJavaScriptError(error: Error): AppError {
    const message = error.message.toLowerCase();
    
    // Network connectivity errors
    if (message.includes('fetch') || message.includes('network') || message.includes('failed to fetch')) {
      return {
        type: 'network',
        message: 'Unable to connect to the server. Please check your internet connection.',
        recoverable: true,
        action: () => {
          // Retry mechanism could be implemented here
        }
      };
    }
    
    // Chunk loading errors (common in React apps)
    if (message.includes('loading chunk') || message.includes('chunkloaderror')) {
      return {
        type: 'client',
        message: 'Failed to load application resources. The page will refresh automatically.',
        recoverable: true,
        action: () => {
          setTimeout(() => window.location.reload(), 2000);
        }
      };
    }
    
    // Authentication errors
    if (message.includes('unauthorized') || message.includes('authentication')) {
      return {
        type: 'auth',
        message: 'Authentication failed. Please log in again.',
        recoverable: true,
        action: () => {
          if (typeof window !== 'undefined') {
            localStorage.removeItem('access_token');
            localStorage.removeItem('user');
            window.location.href = '/login';
          }
        }
      };
    }
    
    // Validation errors
    if (message.includes('validation') || message.includes('invalid')) {
      return {
        type: 'validation',
        message: error.message,
        recoverable: true
      };
    }
    
    // Generic JavaScript error
    return {
      type: 'client',
      message: process.env.NODE_ENV === 'development' 
        ? error.message 
        : 'An unexpected error occurred. Please try again.',
      recoverable: true,
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
    };
  }

  /**
   * Get user-friendly error message with recovery suggestions
   */
  static getErrorMessage(error: AppError): string {
    const baseMessage = error.message;
    
    if (!error.recoverable) {
      return baseMessage;
    }
    
    switch (error.type) {
      case 'network':
        return `${baseMessage} Please check your connection and try again.`;
      case 'auth':
        return `${baseMessage} You will be redirected to the login page.`;
      case 'server':
        return `${baseMessage} If the problem persists, please contact support.`;
      case 'validation':
        return `${baseMessage} Please check your input and try again.`;
      default:
        return baseMessage;
    }
  }

  /**
   * Show appropriate user notification for error
   */
  static showErrorNotification(error: AppError, showToast?: (message: string, type: string) => void) {
    const message = this.getErrorMessage(error);
    
    if (showToast) {
      const toastType = error.type === 'validation' ? 'warning' : 'error';
      showToast(message, toastType);
    } else {
      // Fallback to console in development
      if (process.env.NODE_ENV === 'development') {
        console.error('Error:', error);
      }
    }
    
    // Execute recovery action if available
    if (error.action) {
      error.action();
    }
  }

  /**
   * Retry mechanism for recoverable errors
   */
  static async retryOperation<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    backoffMs: number = 1000
  ): Promise<T> {
    let lastError: any;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        const parsedError = this.parseError(error);
        
        // Don't retry non-recoverable errors
        if (!parsedError.recoverable || attempt === maxRetries) {
          throw error;
        }
        
        // Exponential backoff
        const delay = backoffMs * Math.pow(2, attempt - 1);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError;
  }
}

// Global error handler for unhandled promise rejections
if (typeof window !== 'undefined') {
  window.addEventListener('unhandledrejection', (event) => {
    const error = ErrorHandler.parseError(event.reason);
    console.error('Unhandled promise rejection:', error);
    
    // Prevent default browser behavior for some error types
    if (error.type === 'network' || error.type === 'server') {
      event.preventDefault();
    }
  });
}