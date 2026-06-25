import { createSignal } from 'solid-js';
import type { Message } from '../types/chat';

const [messages, setMessages] = createSignal<Message[]>([]);
const [isLoading, setIsLoading] = createSignal(false);
const [error, setError] = createSignal<string | null>(null);

export const addMessage = (msg: Omit<Message, 'id' | 'timestamp'>) => {
  setMessages(prev => [
    ...prev,
    { ...msg, id: crypto.randomUUID(), timestamp: new Date() },
  ]);
};

export const clearMessages = () => setMessages([]);
export { messages, isLoading, setIsLoading, error, setError };
