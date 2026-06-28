import type { Component } from 'solid-js';
import Header from './components/Header';
import ChatView from './components/ChatView';
import QueryInput from './components/QueryInput';
import { addMessage, setIsLoading, setError } from './stores/chatStore';
import { askQuestion } from './services/queryService';

const handleQuery = async (query: string): Promise<void> => {
  // Add the user's message to the chat immediately for perceived responsiveness -
  // don't wait for the backend to acknowledge before showing it in the thread
  addMessage({ role: 'user', content: query });
  setIsLoading(true);
  setError(null);

  console.log('[DocGraph App] Submitting query:', query);

  try {
    const result = await askQuestion(query);

    // Display the LLM's answer in the chat thread
    addMessage({ role: 'assistant', content: result.answer });

    console.log('[DocGraph App] Answer received. Citations:', result.sources);
    // TODO: Phase 4 - forward result.sources to the PDF viewer panel
    // so the relevant pages are highlighted alongside the answer
  } catch (queryError) {
    console.error('[DocGraph App] Query error:', queryError);
    setError('Failed to reach the backend. Is the server running?');
  } finally {
    setIsLoading(false);
  }
};

const App: Component = () => (
  <div style={{
    'background-color': 'var(--color-bg-primary)',
    color: 'var(--color-text-primary)',
    'min-height': '100vh',
    height: '100vh',
    display: 'flex',
    'flex-direction': 'column',
    transition: 'background-color 0.25s ease, color 0.25s ease',
    'font-family': "'Inter', system-ui, sans-serif",
    overflow: 'hidden',
  }}>
    <Header />
    <div style={{
      flex: '1',
      display: 'flex',
      'flex-direction': 'column',
      width: '100%',
      'max-width': '800px',
      margin: '0 auto',
      'min-height': '0',
      'box-sizing': 'border-box',
    }}>
      <ChatView />
      <QueryInput onSubmit={handleQuery} />
    </div>
  </div>
);

export default App;
