import type { Component } from 'solid-js';
import { theme, toggleTheme } from './stores/themeStore';

const App: Component = () => {
  return (
    <div
      style={{
        'background-color': 'var(--color-bg-primary)',
        color: 'var(--color-text-primary)',
        'min-height': '100vh',
        display: 'flex',
        'flex-direction': 'column',
        'align-items': 'center',
        transition: 'background-color 0.25s ease, color 0.25s ease',
      }}
    >
      {/* Header bar */}
      <div
        style={{
          width: '100%',
          display: 'flex',
          'align-items': 'center',
          'justify-content': 'space-between',
          'border-bottom': '1px solid var(--color-border)',
          padding: '1rem 2rem',
        }}
      >
        <div>
          <h1
            style={{
              'font-size': '1.25rem',
              'font-weight': '600',
              'letter-spacing': '0.05em',
              color: 'var(--color-text-primary)',
              'text-transform': 'uppercase',
            }}
          >
            DocGraph
          </h1>
          <p style={{ 'font-size': '0.75rem', color: 'var(--color-text-muted)', 'margin-top': '0.25rem' }}>
            Medical LLM Assistant · Mistral-7B
          </p>
        </div>

        {/* Theme toggle button */}
        <button
          onClick={toggleTheme}
          style={{
            background: 'var(--color-bg-secondary)',
            border: '1px solid var(--color-border)',
            'border-radius': '2px',
            color: 'var(--color-text-secondary)',
            cursor: 'pointer',
            'font-family': 'inherit',
            'font-size': '0.8rem',
            padding: '0.4rem 0.8rem',
            transition: 'border-color 0.15s ease',
          }}
        >
          {theme() === 'dark' ? '☀ Light' : '● Dark'}
        </button>
      </div>

      {/* Main content area — centered, ready for Phase 2 chat UI */}
      <div
        style={{
          flex: '1',
          width: '100%',
          'max-width': '760px',
          display: 'flex',
          'flex-direction': 'column',
          padding: '2rem',
        }}
      >
        {/* Chat components will be mounted here in Phase 2 */}
      </div>
    </div>
  );
};

export default App;
