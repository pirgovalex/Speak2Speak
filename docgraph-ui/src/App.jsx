import Header from './components/Header';
import ChatView from './components/ChatView';
import QueryInput from './components/QueryInput';
import { addMessage, setIsLoading, setError } from './stores/chatStore';
const handleQuery = async (query) => {
    addMessage({ role: 'user', content: query });
    setIsLoading(true);
    setError(null);
    try {
        // Placeholder — Phase 3 will wire Axios here
        await new Promise(r => setTimeout(r, 1200));
        addMessage({
            role: 'assistant',
            content: '[Phase 3 will connect this to the Mistral-7B backend via Axios]',
        });
    }
    catch {
        setError('Failed to reach the backend.');
    }
    finally {
        setIsLoading(false);
    }
};
const App = () => (<div style={{
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
      <QueryInput onSubmit={handleQuery}/>
    </div>
  </div>);
export default App;
