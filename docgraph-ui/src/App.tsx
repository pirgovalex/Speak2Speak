import type { Component } from 'solid-js';
import { theme, toggleTheme } from './stores/themeStore';

const App: Component = () => {
  return (
    <div
      style={{
        'background-color': 'var(--color-bg-primary)',
        color: 'var(--color-text-primary)',
        'min-height': '100vh',
        padding: '2rem',
        transition: 'background-color 0.25s ease, color 0.25s ease',
      }}
    >
      {/* Header bar */}
      <div
        style={{
          display: 'flex',
          'align-items': 'center',
          'justify-content': 'space-between',
          'border-bottom': '1px solid var(--color-border)',
          'padding-bottom': '1rem',
          'margin-bottom': '2rem',
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

      {/* Phase 1 status card */}
      <div
        style={{
          background: 'var(--color-bg-card)',
          border: '1px solid var(--color-border)',
          'border-radius': '0',
          'box-shadow': 'var(--shadow-card)',
          padding: '1.5rem',
          'max-width': '540px',
        }}
      >
        <p
          style={{
            'font-size': '0.7rem',
            'font-weight': '500',
            'letter-spacing': '0.12em',
            'text-transform': 'uppercase',
            color: 'var(--color-accent)',
            'margin-bottom': '0.75rem',
          }}
        >
          Phase 1 — Scaffold
        </p>
        <p style={{ color: 'var(--color-text-primary)', 'font-size': '0.95rem' }}>
          SolidJS + Vite + TypeScript + TailwindCSS v3
        </p>
        <p style={{ color: 'var(--color-text-secondary)', 'font-size': '0.85rem', 'margin-top': '0.5rem' }}>
          Theme system initialized. Current theme:{' '}
          <span style={{ color: 'var(--color-accent)', 'font-weight': '500' }}>{theme()}</span>
        </p>
        <div
          style={{
            'margin-top': '1.25rem',
            'padding-top': '1.25rem',
            'border-top': '1px solid var(--color-border)',
            display: 'flex',
            gap: '0.5rem',
            'flex-wrap': 'wrap',
          }}
        >
          {['Scaffold ✓', 'Tailwind ✓', 'Theme Store ✓', 'CSS Vars ✓', 'Dark Default ✓'].map(label => (
            <span
              style={{
                background: 'var(--color-accent-subtle)',
                color: 'var(--color-accent)',
                border: '1px solid var(--color-accent)',
                'border-radius': '2px',
                'font-size': '0.7rem',
                'font-weight': '500',
                padding: '0.2rem 0.5rem',
                'letter-spacing': '0.05em',
              }}
            >
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
