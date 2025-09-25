import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';

interface MessageContentProps {
  content: string;
  isUser: boolean;
}

const MessageContent = React.memo(function MessageContent({ content, isUser }: MessageContentProps) {
  console.log('MessageContent rendering:', { isUser, contentLength: content.length, contentPreview: content.substring(0, 50) + '...' });
  
  if (isUser) {
    // User messages don't need markdown processing
    return <div className="whitespace-pre-wrap text-white">{content}</div>;
  }

  // Preprocess content to fix common markdown issues and clean up AI responses
  const preprocessContent = (text: string): string => {
    return text
      // Remove AI thinking tags
      .replace(/<think>[\s\S]*?<\/think>/g, '')
      .replace(/<think>/g, '')
      .replace(/<\/think>/g, '')
      // Remove unnecessary "i.e." patterns that appear in AI responses
      .replace(/\b(i\.e\.\s*)?([A-Z])\s*/g, '$2')
      // Clean up "###Rationale: A i.e." pattern to just "###Rationale:"
      .replace(/###\s*Rationale:\s*[A-Z]\s*(i\.e\.\s*)?/g, '### Rationale: ')
      // Remove standalone letters followed by "i.e."
      .replace(/^[A-Z]\s+i\.e\.\s*/gm, '')
      // Clean up incomplete bullet points ending with "-"
      .replace(/^-\s*$/gm, '')
      .replace(/\n-\s*$/g, '')
      // Fix medical/academic formatting patterns
      .replace(/###\s*Options:\s*/g, '\n### Options:\n')
      .replace(/###\s*Answer:\s*/g, '\n### Answer:\n')
      .replace(/###\s*Explanation:\s*/g, '\n### Explanation:\n')
      .replace(/###\s*Rationale:\s*/g, '\n### Rationale:\n')
      // Fix standalone headings without ###
      .replace(/^Rationale:\s*$/gm, '### Rationale:')
      .replace(/^Answer:\s*$/gm, '### Answer:')
      .replace(/^Options:\s*$/gm, '### Options:')
      .replace(/^Explanation:\s*$/gm, '### Explanation:')
      // Clean up option patterns like "A. " to be proper lists
      .replace(/^([A-D])\.\s+([^\n]+)/gm, '**$1.** $2')
      // Fix common text issues
      .replace(/^he\s+/gm, 'The ')
      .replace(/OPTION\s*([A-D])IS\s*CORRECT/g, 'OPTION $1 IS CORRECT')
      .replace(/OPTION\s*([A-D])\s*IS\s*CORRECT/g, '**The correct answer is Option $1.**')
      // Fix headings with ### followed by numbers (like ### 1., ### 2.)
      .replace(/#{3,}\s*(\d+)\.\s*/g, '\n### $1. ')
      // Fix patterns like "### 1. Make it a Daily Habit"
      .replace(/#{3,}\s*(\d+)\.\s+([^\n]+)/g, '\n### $1. $2\n')
      // Fix asterisks that should be bullet points
      .replace(/^\*\s*([A-Z][^*\n]+):/gm, '* **$1**:')
      // Fix patterns like "*Goal:" to "* **Goal**:"
      .replace(/^\*([A-Za-z]+):/gm, '* **$1**:')
      // Fix patterns like "* *Basic:" to "* **Basic**:"
      .replace(/^\*\s*\*([^*]+):/gm, '* **$1**:')
      // Fix bullet points that start with * and contain **bold**
      .replace(/^\*\s*\*\*([^*]+)\*\*\s*([^\n]*)/gm, '* **$1** $2')
      // Fix patterns like "* **Speaking**:"
      .replace(/^\*\s*\*\*([^*]+)\*\*:\s*/gm, '* **$1**: ')
      // Fix patterns like "* **Speaking** (explanation)"
      .replace(/^\*\s*\*\*([^*]+)\*\*\s*\(([^)]+)\)/gm, '* **$1** ($2)')
      // Fix standalone asterisks at beginning of lines
      .replace(/^\*([^*\s][^*\n]*)/gm, '* $1')
      // Fix Topic labels like "*Topic 1: Arrays & Strings"
      .replace(/^\*Topic\s+(\d+):\s*([^\n]+)/gm, '\n**Topic $1: $2**\n')
      // Fix Level labels like "*Level 1: The Foundation"
      .replace(/^\*Level\s+(\d+):\s*([^\n]+)/gm, '\n**Level $1: $2**\n')
      // Fix Platforms to use labels
      .replace(/^\*Platforms to use:/gm, '\n**Platforms to use:**\n')
      // Clean up medical terminology patterns
      .replace(/\(SIDS\)\s+is\s+defined\s+as\s+the/g, '(SIDS) is defined as the')
      // Fix incomplete sentences ending with conjunctions
      .replace(/\s+(and|or|but)\s*$/gm, '')
      // Remove trailing hyphens and incomplete list items
      .replace(/^-\s*$/gm, '')
      .replace(/\s+-\s*$/gm, '')
      // Ensure proper line breaks before headings
      .replace(/([^\n])(\n*#{1,6}\s+)/g, '$1\n\n$2')
      // Ensure proper line breaks before bullet points
      .replace(/([^\n])(\n*\*\s+)/g, '$1\n$2')
      // Fix multiple spaces
      .replace(/[ ]{2,}/g, ' ')
      // Clean up extra newlines
      .replace(/\n{3,}/g, '\n\n')
      // Trim whitespace
      .trim();
  };

  const processedContent = preprocessContent(content);
  console.log('Processed content preview:', processedContent.substring(0, 100) + '...');

  return (
    <div className="max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // Modern heading styles - cleaner, more spacious like ChatGPT
          h1: ({ children }) => (
            <h1 className="text-xl font-semibold mb-4 mt-6 text-gray-900 first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-semibold mb-3 mt-5 text-gray-900 first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold mb-3 mt-4 text-gray-900 first:mt-0">{children}</h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-base font-medium mb-2 mt-3 text-gray-800 first:mt-0">{children}</h4>
          ),
          // Modern paragraph styles - better line height and spacing
          p: ({ children }) => (
            <p className="mb-4 text-gray-800 leading-7 text-[15px]">{children}</p>
          ),
          // Better list styles - more modern spacing
          ul: ({ children }) => (
            <ul className="mb-4 text-gray-800 space-y-1 pl-0">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-4 text-gray-800 space-y-1 pl-0 list-decimal list-inside">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="text-gray-800 leading-7 text-[15px] flex items-start pl-0 mb-1">
              <span className="text-gray-400 mr-2 mt-[2px] flex-shrink-0">•</span>
              <span className="flex-1">{children}</span>
            </li>
          ),
          // Modern code styles - more like ChatGPT
          code: ({ children, className, ...props }: any) => {
            const isInline = !className?.includes('language-');
            if (isInline) {
              return (
                <code className="bg-gray-100 text-gray-800 px-1.5 py-0.5 rounded text-sm font-mono border">
                  {children}
                </code>
              );
            }
            return (
              <pre className="bg-[#0d1117] text-gray-100 p-4 rounded-lg text-sm font-mono overflow-x-auto mb-4 border border-gray-700">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            );
          },
          pre: ({ children }) => (
            <div className="mb-4">
              {children}
            </div>
          ),
          // Modern blockquote styles
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-gray-300 pl-4 py-1 mb-4 bg-gray-50 text-gray-700 rounded-r text-[15px]">
              {children}
            </blockquote>
          ),
          // Better strong/bold text
          strong: ({ children }) => (
            <strong className="font-semibold text-gray-900">{children}</strong>
          ),
          // Better emphasis/italic text
          em: ({ children }) => (
            <em className="italic text-gray-700">{children}</em>
          ),
          // Customize links
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-blue-600 hover:text-blue-800 underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              {children}
            </a>
          ),
          // Modern table styles
          table: ({ children }) => (
            <div className="overflow-x-auto mb-4">
              <table className="min-w-full border border-gray-200 rounded-lg text-[15px]">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-gray-50">
              {children}
            </thead>
          ),
          th: ({ children }) => (
            <th className="border border-gray-200 px-3 py-3 text-left font-semibold text-gray-900 text-[15px]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-gray-200 px-3 py-3 text-gray-800 text-[15px] leading-6">
              {children}
            </td>
          ),
          // Subtle horizontal rule
          hr: () => (
            <hr className="my-6 border-gray-200" />
          ),
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
});

export default MessageContent;
