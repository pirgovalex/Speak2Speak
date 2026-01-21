import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from LLM import llama_interact
from tts import speak
from load_pdf import load_and_store_pdf
import database_manager
import threading
import whisper
import sounddevice as sd
import wave
import tempfile
import os
import uuid

# --- Global Whisper Model ---
whisper_model = None
model_status = "Loading specific features..."

# --- Global Thread State ---
current_thread_id = "default_user_thread"

def load_whisper_model():
    global whisper_model, model_status
    print("Loading Whisper model...")
    try:
        # 'base' is a good balance. 'medium' are better for medical terms but slower.
        whisper_model = whisper.load_model("base")
        model_status = "Ready"
        print("Whisper model loaded successfully.")
        # Update UI if root exists
        if 'btn_speech' in globals(): # Check global vars.
             btn_speech.config(state='normal', text="Speak")
    except Exception as e:
        model_status = "Error loading speech model"
        print(f"Error loading Whisper: {e}")

# Start loading model in background
threading.Thread(target=load_whisper_model, daemon=True).start()

def update_thread_dropdown():
    threads = database_manager.get_recent_threads()
    if not threads:
        threads = ["default_user_thread"]
    
    # Update combobox values
    thread_combo['values'] = threads
    
    # If current thread is not in values (e.g. new custom one), add it or select it
    if current_thread_id not in threads:
         current_threads = list(thread_combo['values'])
         current_threads.insert(0, current_thread_id)
         thread_combo['values'] = current_threads
    
    thread_combo.set(current_thread_id)

def on_new_chat():
    global current_thread_id
    # Create a new unique thread ID
    current_thread_id = str(uuid.uuid4())[:8] # Short UUID for readability
    update_thread_dropdown()
    reset_text_area()
    print(f"Started new chat: {current_thread_id}")

def load_history_to_ui(thread_id):
    history = database_manager.get_chat_history(thread_id)
    txt_area.config(state='normal')
    txt_area.delete(1.0, tk.END)
    
    if not history:
        txt_area.insert(tk.END, f"--- Conversation: {thread_id} ---\n")
    
    for msg in history:
        sender = msg['sender'].capitalize()
        content = msg['content']
        txt_area.insert(tk.END, f"\n{sender}: {content}\n")
        txt_area.insert(tk.END, "-"*20 + "\n")
        
    txt_area.see(tk.END)
    txt_area.config(state='disabled')

def on_thread_select(event):
    global current_thread_id
    selected = thread_combo.get()
    if selected and selected != current_thread_id:
        current_thread_id = selected
        print(f"Switched to thread: {current_thread_id}")
        load_history_to_ui(current_thread_id)

def on_click():
    def task():
        btn_create_vector_db.config(text="Processing...", state="disabled", bg="gray")
        try:
            load_and_store_pdf()
            btn_create_vector_db.config(text="SAVED", state="normal", bg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to store PDF:\n{e}")
            btn_create_vector_db.config(text="STORE PDF", state="normal", bg="maroon")
            
    threading.Thread(target=task, daemon=True).start()

def ask_question():
    user_q = entry.get()
    if not user_q.strip():
        messagebox.showwarning("Warning", "Please enter a question!")
        return
    def worker():
    # call LLM function with current thread_id
        cleaned = llama_interact(user_q, thread_id=current_thread_id)

        txt_area.config(state='normal')
        # txt_area.delete(1.0, tk.END) # Don't clear history for this session view
        txt_area.insert(tk.END, f"\nUser: {user_q}\nAI: {cleaned}\n")
        txt_area.insert(tk.END, "-"*20 + "\n")
        txt_area.see(tk.END) # Scroll to bottom
        txt_area.config(state='disabled')
        
        # Clear entry
        entry.delete(0, tk.END)

        speak(cleaned)
    threading.Thread(target=worker,daemon=True).start()

def speech_to_text():
    global whisper_model
    
    if whisper_model is None:
        messagebox.showinfo("Please Wait", "Speech model is still loading...")
        return

    def record_audio(duration=5, fs=16000):
        # Notify user (visual cue preferred over blocking popup, but simplified here)
        btn_speech.config(text="Recording...", bg="#ffcccc")
        root.update()
        
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        
        btn_speech.config(text="Processing...", bg="#e6e6e6")
        root.update()
        return recording, fs

    def process_audio():
        try:
            # Record
            audio, fs = record_audio(duration=6)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(audio.tobytes())
                audio_path = f.name

            # Transcribe
            # Removed hardcoded language="en" as bugfix
            # Relaxed thresholds
            result = whisper_model.transcribe(
                audio_path, 
                fp16=False,
                # temperature=0.0, # Default is usually fine
                # no_speech_threshold=0.6, # Default if hallucinations come back try this
                # logprob_threshold=-1.0 # Default
            )
            
            query = result.get("text", "").strip()
            segments = result.get("segments", [])
            
            # Logging for debugging
            if segments:
                avg_logprob = sum(seg["avg_logprob"] for seg in segments) / len(segments)
                print(f"Transcription: '{query}' | Avg Logprob: {avg_logprob:.3f} | Language: {result.get('language')}")
            else:
                print("No segments found.")

            # Cleanup
            os.unlink(audio_path)
            
            # Reset button
            btn_speech.config(text="Speak", bg="#f0f0f0")

            # Basic hallucination filter
            hallucinations = ["Subtitles by", "The Snow", "Amara.org", "Community", "Copyright"]
            if any(h.lower() in query.lower() for h in hallucinations):
                 print(f"Filtered hallucination: {query}")
                 query = ""

            if query:
                # Update GUI safely
                entry.delete(0, tk.END)
                entry.insert(0, query)
                print(f"You said: {query}")
                # Auto-submit if spoken? Optional. Let's keep manual submit for now.
            else:
                print("No valid speech detected.")

        except Exception as e:
            print(f"Speech Error: {e}")
            messagebox.showerror("Error", f"Speech recognition failed:\n{str(e)}")
            btn_speech.config(text="Speak", bg="SystemButtonFace")

    # Run recording/processing in a thread so GUI doesn't freeze
    # Note: sd.rec and sd.wait are blocking, so doing this in a thread is good,
    # but we need to be careful about Tkinter thread safety. Multithreading in python is not real :) 
    # For this "proof of concept", threading the whole thing excluding the final UI update is safest.
    threading.Thread(target=process_audio, daemon=True).start()

# GUI
root = tk.Tk()
root.title("Medical LLM Assistant")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Row 0: Chat Context Controls
lbl_thread = tk.Label(frame, text="Chat ID:")
lbl_thread.grid(row=0, column=0, padx=5)

thread_combo = ttk.Combobox(frame, width=20, state="readonly")
thread_combo.grid(row=0, column=1, padx=5)
thread_combo.bind("<<ComboboxSelected>>", on_thread_select)

btn_new_chat = tk.Button(frame, text="New Chat", command=on_new_chat)
btn_new_chat.grid(row=0, column=2, padx=5)

# Row 1: Input and standard controls
entry = tk.Entry(frame, width=60)
entry.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

btn_ask = tk.Button(frame, text="Ask", command=ask_question)
btn_ask.grid(row=1, column=2, padx=5)

btn_speech = tk.Button(frame, text="Speak (Loading...)", command=speech_to_text, state='disabled')
btn_speech.grid(row=1, column=3, padx=5)

btn_reset = tk.Button(frame, text="Clear Screen", command=lambda: reset_text_area())
btn_reset.grid(row=1, column=4, padx=5)

# Row 2: Admin controls
btn_create_vector_db = tk.Button(frame,
                                    text="STORE PDF",
                                    bg="maroon",
                                    fg="white",
                                    activebackground="maroon",
                                    activeforeground="gray",
                                    font=("Segoe UI", 11, "bold"),
                                    relief="raised",
                                    bd=3,
                                    command=on_click
                                    )
btn_create_vector_db.grid(row=2, column=0, columnspan=5, padx=5, pady=10)


txt_area = scrolledtext.ScrolledText(root, width=100, height=20, state='disabled', wrap='word')
txt_area.pack(padx=10, pady=10)

def reset_text_area():
    txt_area.config(state='normal')
    txt_area.delete(1.0, tk.END)
    txt_area.config(state='disabled')

# Initialize thread dropdown
update_thread_dropdown()

if __name__ == '__main__':
    root.mainloop()