import tkinter as tk
from tkinter import scrolledtext, messagebox
import speech_recognition as sr
from LLM import llama_interact
from tts import speak
import  threading

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
    r = sr.Recognizer()
    with sr.Microphone() as source:
        messagebox.showinfo("Info", "Listening... Please speak clearly.")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        entry.delete(0, tk.END)
        entry.insert(0, query)
    except Exception as e:
        messagebox.showerror("Error", f"Could not recognize speech: {e}")

# GUI
root = tk.Tk()
root.title("Medical LLM Assistant")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

entry = tk.Entry(frame, width=80)
entry.grid(row=0, column=0, padx=5, pady=5)

btn_ask = tk.Button(frame, text="Ask", command=ask_question)
btn_ask.grid(row=0, column=1, padx=5)

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
root.mainloop()
