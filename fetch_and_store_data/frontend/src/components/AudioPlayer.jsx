import { createSignal, onMount, onCleanup } from 'solid-js';
import { pauseTTS, resumeTTS, stopTTS } from '../lib/api';
import { Play, Pause, Square } from 'lucide-solid';

// Global signal for player state
export const [isPlaying, setIsPlaying] = createSignal(false);
export const [showPlayer, setShowPlayer] = createSignal(false);

export default function AudioPlayer() {
    const togglePlay = async () => {
        if (isPlaying()) {
            await pauseTTS();
            setIsPlaying(false);
        } else {
            await resumeTTS();
            setIsPlaying(true);
        }
    };

    const handleStop = async () => {
        await stopTTS();
        setIsPlaying(false);
        setShowPlayer(false);
    };

    return (
        <>
            {showPlayer() && (
                <div class="fixed bottom-4 right-4 bg-white shadow-lg border border-slate-200 rounded-lg p-3 flex items-center gap-3 z-50 animate-in fade-in slide-in-from-bottom-4">
                    <div class="text-sm font-medium text-slate-700 mr-2">
                        Playing Audio...
                    </div>

                    <button
                        onClick={togglePlay}
                        class="btn btn-ghost p-2 rounded-full hover:bg-slate-100"
                        title={isPlaying() ? "Pause" : "Resume"}
                    >
                        {isPlaying() ? <Pause size={20} /> : <Play size={20} />}
                    </button>

                    <button
                        onClick={handleStop}
                        class="btn btn-ghost p-2 rounded-full hover:bg-red-50 text-red-500"
                        title="Stop"
                    >
                        <Square size={20} />
                    </button>
                </div>
            )}
        </>
    );
}
