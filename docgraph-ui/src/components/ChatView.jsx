import { For, Show, createEffect } from 'solid-js';
import { messages, isLoading } from '../stores/chatStore';
import MessageBubble from './MessageBubble';
import LoadingIndicator from './LoadingIndicator';
const ChatView = () => {
    let bottomRef;
    const scrollToBottom = () => {
        bottomRef?.scrollIntoView({ behavior: 'smooth' });
    };
    createEffect(() => {
        // Track both reactive sources so the effect re-runs on either change
        messages();
        isLoading();
        scrollToBottom();
    });
    return (<div style={{
            flex: '1',
            'overflow-y': 'auto',
            padding: '1.25rem 1.5rem',
            display: 'flex',
            'flex-direction': 'column',
            'min-height': '0',
            'box-sizing': 'border-box',
        }}>
      <Show when={messages().length > 0} fallback={<div style={{
                flex: '1',
                display: 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                color: 'var(--color-text-muted)',
                'font-size': '0.85rem',
                gap: '0.5rem',
            }}>
            <span style={{ 'font-size': '1.5rem', opacity: '0.4' }}>⊕</span>
            <p style={{ margin: '0', 'letter-spacing': '0.03em' }}>Ask a medical question to begin</p>
          </div>}>
        <For each={messages()}>
          {(msg) => <MessageBubble message={msg}/>}
        </For>
        <Show when={isLoading()}>
          <LoadingIndicator />
        </Show>
      </Show>
      <div ref={bottomRef}/>
    </div>);
};
export default ChatView;
