import { createSignal, type Component } from 'solid-js';
import { theme, toggleTheme } from '../stores/themeStore';
import { uploadPdf } from '../services/documentService';

const Header: Component = () => {
  const [isUploading, setIsUploading] = createSignal(false);
  let fileInputRef: HTMLInputElement | undefined;

  const handleFileChange = async (e: Event) => {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      setIsUploading(true);
      try {
        await uploadPdf(target.files[0]);
      } catch (err) {
        console.error(err);
      } finally {
        setIsUploading(false);
        target.value = '';
      }
    }
  };

  return (
  <header style={{
    width: '100%',
    display: 'flex',
    'align-items': 'center',
    'justify-content': 'space-between',
    'border-bottom': '1px solid var(--color-border)',
    padding: '0.875rem 1.5rem',
    background: 'var(--color-bg-primary)',
    position: 'sticky',
    top: '0',
    'z-index': '10',
    'box-sizing': 'border-box',
  }}>
    <div>
      <h1 style={{
        'font-size': '1rem',
        'font-weight': '600',
        'letter-spacing': '0.08em',
        'text-transform': 'uppercase',
        color: 'var(--color-text-primary)',
        margin: '0',
      }}>DocGraph</h1>
      <p style={{
        'font-size': '0.7rem',
        color: 'var(--color-text-muted)',
        margin: '0.15rem 0 0',
      }}>Medical LLM Assistant</p>
    </div>
    <div style={{ display: 'flex', gap: '0.5rem', 'align-items': 'center' }}>
      <input type="file" accept="application/pdf" style={{ display: 'none' }} ref={fileInputRef} onChange={handleFileChange} />
      <button
        onClick={() => fileInputRef?.click()}
        disabled={isUploading()}
        style={{
          background: 'var(--color-accent)',
          border: '1px solid var(--color-accent)',
          'border-radius': '2px',
          color: 'var(--color-bg-primary)',
          cursor: isUploading() ? 'wait' : 'pointer',
          'font-family': 'inherit',
          'font-size': '0.78rem',
          padding: '0.35rem 0.75rem',
          opacity: isUploading() ? '0.7' : '1',
          transition: 'opacity 0.2s ease',
        }}
      >
        {isUploading() ? 'uploading...' : 'upload pdf'}
      </button>
      <button
        id="theme-toggle-btn"
        onClick={toggleTheme}
        style={{
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
          'border-radius': '2px',
          color: 'var(--color-text-secondary)',
          cursor: 'pointer',
          'font-family': 'inherit',
          'font-size': '0.78rem',
          padding: '0.35rem 0.75rem',
          transition: 'border-color 0.15s ease',
        }}
      >
        {theme() === 'dark' ? '☀ Light' : '● Dark'}
      </button>
    </div>
  </header>
  );
};

export default Header;
