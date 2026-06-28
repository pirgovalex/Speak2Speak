import type { Component } from 'solid-js';
import type { Message } from '../types/chat';

interface Props { message: Message; }

const MessageBubble: Component<Props> = (props) => {
  const isUser = () => props.message.role === 'user';

  const time = () => props.message.timestamp.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit',
  });

  return (
    <div style={{
      display: 'flex',
      'justify-content': isUser() ? 'flex-end' : 'flex-start',
      'margin-bottom': '0.75rem',
      'align-items': 'flex-end',
      gap: '0.5rem',
    }}>
      {/* Role indicator for assistant */}
      {!isUser() && (
        <div style={{
          'font-size': '0.65rem',
          color: 'var(--color-text-muted)',
          'font-weight': '500',
          'letter-spacing': '0.06em',
          'text-transform': 'uppercase',
          'padding-bottom': '0.25rem',
          'min-width': '2rem',
          'text-align': 'center',
        }}>AI</div>
      )}

      <div style={{
        'max-width': '72%',
        padding: '0.6rem 0.9rem',
        background: isUser() ? 'var(--color-user-bubble)' : 'var(--color-ai-bubble)',
        color: isUser() ? 'var(--color-user-bubble-text)' : 'var(--color-ai-bubble-text)',
        'border-radius': '2px',
        'font-size': '0.875rem',
        'line-height': '1.55',
        'border-left': !isUser() ? '2px solid var(--color-accent)' : 'none',
        'word-break': 'break-word',
      }}>
        {/* textContent, NOT innerHTML - XSS prevention */}
        <p style={{ margin: '0', 'white-space': 'pre-wrap' }}>{props.message.content}</p>
        <span style={{
          display: 'block',
          'font-size': '0.65rem',
          color: isUser() ? 'var(--color-user-bubble-text)' : 'var(--color-text-muted)',
          opacity: '0.7',
          'margin-top': '0.3rem',
          'text-align': isUser() ? 'right' : 'left',
        }}>{time()}</span>
      </div>

      {isUser() && (
        <div style={{
          'font-size': '0.65rem',
          color: 'var(--color-text-muted)',
          'font-weight': '500',
          'letter-spacing': '0.06em',
          'text-transform': 'uppercase',
          'padding-bottom': '0.25rem',
          'min-width': '2rem',
          'text-align': 'center',
        }}>You</div>
      )}
    </div>
  );
};

export default MessageBubble;
