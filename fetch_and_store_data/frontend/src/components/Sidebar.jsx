import { createSignal, onMount, For } from 'solid-js';
import { getThreads, createThread, deleteThread } from '../lib/api';
import { A, useNavigate, useParams } from '@solidjs/router';
import { MessageSquarePlus, MessageSquare, Trash2 } from 'lucide-solid';

export default function Sidebar(props) {
    const [threads, setThreads] = createSignal([]);
    const navigate = useNavigate();
    const params = useParams();

    const loadThreads = async () => {
        try {
            const list = await getThreads();

            // If list is empty, create one
            if (list.length === 0) {
                handleNewChat();
                return;
            }
            setThreads(list);
        } catch (e) {
            console.error(e);
        }
    };

    const handleNewChat = async () => {
        try {
            const res = await createThread();
            setThreads([res.id, ...threads()]);
            navigate(`/chat/${res.id}`);
        } catch (e) {
            console.error(e);
        }
    };

    const handleDelete = async (e, id) => {
        e.preventDefault();
        e.stopPropagation();
        if (!confirm('Are you sure you want to delete this chat?')) return;

        try {
            await deleteThread(id);
            setThreads(threads().filter(t => t !== id));
            // If we deleted the current chat, navigate away or to new
            if (params.id === id) {
                if (threads().length > 0) {
                    navigate(`/chat/${threads()[0]}`);
                } else {
                    handleNewChat();
                }
            }
        } catch (err) {
            console.error('Failed to delete:', err);
        }
    };

    onMount(loadThreads);

    return (
        <div class="sidebar">
            <div class="p-4 border-b border-slate-200">
                <button
                    onClick={handleNewChat}
                    class="btn btn-primary w-full justify-center"
                >
                    <MessageSquarePlus size={20} />
                    New Chat
                </button>
            </div>

            <div class="flex-1 overflow-y-auto p-2">
                <div class="text-sm font-bold text-secondary mb-2 px-2 mt-2">Recent Chats</div>
                <div class="flex flex-col gap-1">
                    <For each={threads()}>
                        {(id) => (
                            <div class="group relative flex items-center">
                                <A
                                    href={`/chat/${id}`}
                                    class="btn btn-ghost justify-start text-sm w-full truncate pr-10"
                                    classList={{
                                        "bg-slate-100": params.id === id
                                    }}
                                >
                                    <MessageSquare size={16} class="text-secondary" />
                                    <span class="truncate">Thread {id}</span>
                                </A>
                                <button
                                    onClick={(e) => handleDelete(e, id)}
                                    class="absolute right-1 p-1 text-slate-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                                    title="Delete Chat"
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        )}
                    </For>
                </div>
            </div>
        </div>
    );
}
