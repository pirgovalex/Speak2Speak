import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock

# mock modules
mock_tk = MagicMock()
mock_tk.END = "end"
sys.modules["tkinter"] = mock_tk
sys.modules["tkinter.scrolledtext"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()

sys.modules["LLM"] = MagicMock()
sys.modules["tts"] = MagicMock()
sys.modules["load_pdf"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["faster_whisper"] = MagicMock()

import gui

# replace speech_to_text inside gui with print statements or just replace the entry
original_entry = gui.entry
original_messagebox = gui.messagebox

def fake_delete(*args, **kwargs):
    print("DELETE CALLED", args)

def fake_insert(*args, **kwargs):
    print("INSERT CALLED", args)

def fake_warning(*args, **kwargs):
    print("WARNING CALLED", args)

gui.entry.delete = fake_delete
gui.entry.insert = fake_insert
gui.messagebox.showwarning = fake_warning

# setup mocks
mock_sd = sys.modules["sounddevice"]
mock_sd.rec.return_value = "fake_audio"

mock_model_instance = MagicMock()
gui.WhisperModel = MagicMock(return_value=mock_model_instance)

mock_segment = MagicMock()
mock_segment.text = "hello world"
mock_model_instance.transcribe.return_value = ([mock_segment], None)

try:
    print("Calling speech_to_text...")
    gui.speech_to_text()
    print("Function done.")
except Exception as e:
    import traceback
    traceback.print_exc()
