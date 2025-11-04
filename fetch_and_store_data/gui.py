import tkinter as tk
from tkinter import scrolledtext, messagebox
from LLM import llama_interact
from tts import speak
from  load_pdf import load_and_store_pdf
import  threading
import os

def on_click():
    threading.Thread(target=load_and_store_pdf, daemon=True).start()

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
    import whisper
    import sounddevice as sd
    import wave
    import tempfile
    import os

    def record_audio(duration=5, fs=16000):
        messagebox.showinfo("Recording", f"Speak now... ({duration} seconds)")
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        return recording, fs

    try:
        # Load model (first time = download)
        model = whisper.load_model("tiny")  # fast, ~75MB

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
        result = model.transcribe(audio_path, language="en", fp16=False)
        query = result["text"].strip()

        # Cleanup
        os.unlink(audio_path)

        if query:
            entry.delete(0, tk.END)
            entry.insert(0, query)
            print(f"You said: {query}")
        else:
            messagebox.showwarning("No Speech", "Nothing was heard. Try again.")

    except Exception as e:
        messagebox.showerror("Error", f"Speech recognition failed:\n{str(e)}")

# GUI
root = tk.Tk()
root.title("Medical LLM Assistant")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

entry = tk.Entry(frame, width=80)
entry.grid(row=0, column=0, padx=5, pady=5)

btn_ask = tk.Button(frame, text="Ask", command=ask_question)
btn_ask.grid(row=0, column=1, padx=5)

if "faiss_index" not in os.listdir():

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


btn_speech = tk.Button(frame, text="Speak", command=speech_to_text)
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