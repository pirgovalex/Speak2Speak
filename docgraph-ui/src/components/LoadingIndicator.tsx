import type { Component } from 'solid-js';
import { onMount } from 'solid-js';

const LoadingIndicator: Component = () => {
  // Inject keyframes once into <head>
  onMount(() => {
    if (!document.getElementById('docgraph-pulse-style')) {
      const style = document.createElement('style');
      style.id = 'docgraph-pulse-style';
      style.textContent = `
        @keyframes dg-pulse {
          0%, 80%, 100% { opacity: 0.15; transform: scale(0.85); }
          40%            { opacity: 1;    transform: scale(1);    }
        }
        .dg-dot {
          display: inline-block;
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--color-text-muted);
          animation: dg-pulse 1.2s infinite ease-in-out;
        }
        .dg-dot:nth-child(2) { animation-delay: 0.2s; }
        .dg-dot:nth-child(3) { animation-delay: 0.4s; }
      `;
      document.head.appendChild(style);
    }
  });

  return (
    <div style={{
      display: 'flex',
      'justify-content': 'flex-start',
      'margin-bottom': '0.75rem',
      'align-items': 'flex-end',
      gap: '0.5rem',
    }}>
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
      <div style={{
        padding: '0.6rem 0.9rem',
        background: 'var(--color-ai-bubble)',
        'border-radius': '2px',
        'border-left': '2px solid var(--color-accent)',
        display: 'flex',
        gap: '5px',
        'align-items': 'center',
      }}>
        <span class="dg-dot" />
        <span class="dg-dot" />
        <span class="dg-dot" />
      </div>
    </div>
  );
};

export default LoadingIndicator;
