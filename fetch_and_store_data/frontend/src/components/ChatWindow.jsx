import { createSignal, createEffect, onMount, For } from 'solid-js';
import { useParams } from '@solidjs/router';
import { getHistory, sendMessage, speakText } from '../lib/api';
import { Send, Bot, User, Volume2 } from 'lucide-solid';

export default function ChatWindow() {
    const params = useParams();
    const [messages, setMessages] = createSignal([]);
    const [input, setInput] = createSignal("");
    const [loading, setLoading] = createSignal(false);
    let bottomRef;

    const loadMessages = async () => {
        if (!params.id) return;
        try {
            const hist = await getHistory(params.id);
            setMessages(hist);
        } catch (e) {
            console.error(e);
        }
    };

    createEffect(() => {
        // Reload when ID changes
        loadMessages();
    });

    createEffect(() => {
        // Auto scroll
        messages();
        if (bottomRef) bottomRef.scrollIntoView({ behavior: 'smooth' });
    });

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input().trim() || loading()) return;

        const text = input();
        setInput("");

        // Optimistic update
        const userMsg = { sender: 'user', content: text, timestamp: new Date().toISOString() };
        setMessages([...messages(), userMsg]);
        setLoading(true);

        try {
            const res = await sendMessage(params.id, text);
            const aiMsg = { sender: 'ai', content: res.response, timestamp: new Date().toISOString() };
            setMessages([...messages(), aiMsg]);
        } catch (e) {
            console.error(e);
            // Simplify error handling
            setMessages([...messages(), { sender: 'system', content: "Error sending message.", timestamp: "" }]);
        } finally {
            setLoading(false);
        }
    };

    const handleSpeak = async (text) => {
        try {
            // Import these from AudioPlayer dynamically or passed as props? 
            // Better to import directly since signals are exported
            const { setIsPlaying, setShowPlayer } = await import('./AudioPlayer');

            await speakText(text);
            setIsPlaying(true);
            setShowPlayer(true);
        } catch (e) {
            console.error("TTS Failed:", e);
        }
    };

    return (
        <div class="chat-main">
            <div class="p-4 border-b border-slate-200 bg-white flex items-center justify-between">
                <div class="font-bold text-lg">Chat {params.id}</div>
                <div class="text-sm text-secondary">Speak2Speak AI</div>
            </div>

            <div class="message-list">
                <For each={messages()}>
                    {(msg) => (
                        <div class={`message ${msg.sender === 'user' ? 'msg-user' : 'msg-ai'}`}>
                            <div class="text-xs opacity-70 mb-1 flex items-center gap-1 justify-between">
                                <div class="flex items-center gap-1">
                                    {msg.sender === 'user' ? <User size={12} /> : <Bot size={12} />}
                                    {msg.sender === 'user' ? "You" : "Assistant"}
                                </div>
                                {msg.sender === 'ai' && (
                                    <button
                                        onClick={() => handleSpeak(msg.content)}
                                        class="p-1 hover:text-primary transition-colors"
                                        title="Speak"
                                    >
                                        <Volume2 size={14} />
                                    </button>
                                )}
                            </div>
                            <div class="whitespace-pre-wrap">{msg.content}</div>
                        </div>
                    )}
                </For>
                {loading() && (
                    <div class="message msg-ai opacity-70">
                        <div class="flex items-center gap-2">
                            <div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                            <div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                            <div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                        </div>
                    </div>
                )}
                <div ref={bottomRef}></div>
            </div>

            <div class="p-4 bg-white border-t border-slate-200">
                <form onSubmit={handleSubmit} class="flex gap-2 max-w-4xl mx-auto">
                    <input
                        type="text"
                        class="input"
                        value={input()}
                        onInput={(e) => setInput(e.currentTarget.value)}
                        placeholder="Type a message..."
                        disabled={loading()}
                    />
                    <button type="submit" class="btn btn-primary" disabled={loading() || !input().trim()}>
                        <Send size={20} />
                    </button>
                </form>
            </div>
        </div>
    );
}
