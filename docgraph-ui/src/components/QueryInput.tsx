import type { Component } from 'solid-js';
import { createSignal } from 'solid-js';

interface Props {
  onSubmit: (query: string) => void;
  disabled?: boolean;
}

const QueryInput: Component<Props> = (props) => {
  const [value, setValue] = createSignal('');

  const submit = () => {
    const q = value().trim();
    if (!q || props.disabled) return;
    props.onSubmit(q);
    setValue('');
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const isSendDisabled = () => props.disabled || !value().trim();

  return (
    <div style={{
      'border-top': '1px solid var(--color-border)',
      padding: '1rem 1.5rem',
      background: 'var(--color-bg-primary)',
      display: 'flex',
      gap: '0.75rem',
      'align-items': 'flex-end',
      'box-sizing': 'border-box',
    }}>
      <textarea
        id="query-input"
        value={value()}
        onInput={e => setValue(e.currentTarget.value)}
        onKeyDown={onKeyDown}
        disabled={props.disabled}
        placeholder="Ask a medical question..."
        rows={1}
        style={{
          flex: '1',
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
          'border-radius': '2px',
          color: 'var(--color-text-primary)',
          'font-family': 'inherit',
          'font-size': '0.875rem',
          padding: '0.6rem 0.75rem',
          resize: 'none',
          outline: 'none',
          transition: 'border-color 0.15s ease',
          'min-height': '2.5rem',
          'max-height': '8rem',
          overflow: 'auto',
          'line-height': '1.5',
          'box-sizing': 'border-box',
        }}
      />
      <button
        id="send-btn"
        onClick={submit}
        disabled={isSendDisabled()}
        style={{
          background: isSendDisabled()
            ? 'var(--color-bg-secondary)'
            : 'var(--color-accent)',
          border: '1px solid var(--color-border)',
          'border-radius': '2px',
          color: isSendDisabled()
            ? 'var(--color-text-muted)'
            : '#ffffff',
          cursor: isSendDisabled() ? 'not-allowed' : 'pointer',
          'font-family': 'inherit',
          'font-size': '0.85rem',
          'font-weight': '500',
          padding: '0.6rem 1.25rem',
          transition: 'background 0.15s ease, color 0.15s ease',
          'min-width': '5rem',
          height: '2.5rem',
          'white-space': 'nowrap',
        }}
      >
        Send
      </button>
    </div>
  );
};

export default QueryInput;
