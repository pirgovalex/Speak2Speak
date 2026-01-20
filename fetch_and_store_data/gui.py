import tkinter as tk
from tkinter import scrolledtext, messagebox
from LLM import llama_interact
from tts import speak
from load_pdf import load_and_store_pdf
import threading
import whisper
import sounddevice as sd
import wave
import tempfile
import os

# --- Global Whisper Model ---
whisper_model = None
model_status = "Loading specific features..."

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
    # call LLM function
        cleaned = llama_interact(user_q)

        txt_area.config(state='normal')
        txt_area.delete(1.0, tk.END)
        txt_area.insert(tk.END, cleaned)
        txt_area.config(state='disabled')

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

entry = tk.Entry(frame, width=80)
entry.grid(row=0, column=0, padx=5, pady=5)

btn_ask = tk.Button(frame, text="Ask", command=ask_question)
btn_ask.grid(row=0, column=1, padx=5)

# if "faiss_index" not in os.listdir(): 
# Always show the button so user can retry/update

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
btn_create_vector_db.grid(row=0, column=4, padx=5, pady=10,)


btn_speech = tk.Button(frame, text="Speak (Loading...)", command=speech_to_text, state='disabled')
btn_speech.grid(row=0, column=2, padx=5)

txt_area = scrolledtext.ScrolledText(root, width=100, height=20, state='disabled', wrap='word')
txt_area.pack(padx=10, pady=10)

def reset_text_area():
    txt_area.config(state='normal')
    txt_area.delete(1.0, tk.END)
    txt_area.config(state='disabled')

btn_reset = tk.Button(frame, text="Reset", command=reset_text_area)
btn_reset.grid(row=0, column=3, padx=5)

if __name__ == '__main__':
    root.mainloop()