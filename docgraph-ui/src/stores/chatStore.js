import { createSignal } from 'solid-js';
const [messages, setMessages] = createSignal([]);
const [isLoading, setIsLoading] = createSignal(false);
const [error, setError] = createSignal(null);
export const addMessage = (msg) => {
    setMessages(prev => [
        ...prev,
        { ...msg, id: crypto.randomUUID(), timestamp: new Date() },
    ]);
};
export const clearMessages = () => setMessages([]);
export { messages, isLoading, setIsLoading, error, setError };
