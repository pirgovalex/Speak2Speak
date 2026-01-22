const API_BASE = 'http://localhost:8000';

export async function createThread() {
    const res = await fetch(`${API_BASE}/threads/new`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to create thread');
    return res.json();
}

export async function getThreads() {
    const res = await fetch(`${API_BASE}/threads`);
    if (!res.ok) throw new Error('Failed to fetch threads');
    return res.json();
}

export async function speakText(text) {
    const res = await fetch(`${API_BASE}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error('Failed to trigger TTS');
    return res.json();
}

export async function stopTTS() {
    await fetch(`${API_BASE}/tts/stop`, { method: 'POST' });
}

export async function pauseTTS() {
    await fetch(`${API_BASE}/tts/pause`, { method: 'POST' });
}

export async function resumeTTS() {
    await fetch(`${API_BASE}/tts/resume`, { method: 'POST' });
}

export async function deleteThread(threadId) {
    const res = await fetch(`${API_BASE}/threads/${threadId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete thread');
    return res.json();
}

export async function getHistory(threadId) {
    const res = await fetch(`${API_BASE}/threads/${threadId}/history`);
    if (!res.ok) throw new Error('Failed to fetch history');
    return res.json();
}

export async function sendMessage(threadId, message) {
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, message })
    });
    if (!res.ok) throw new Error('Failed to send message');
    return res.json();
}
