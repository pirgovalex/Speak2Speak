import os
import sys
import tkinter as tk
from unittest.mock import MagicMock
import pytest

# add parent directory to path to import gui
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# mock heavy modules before importing gui
# this prevents loading actual models and missing dependencies
sys.modules['LLM'] = MagicMock()
sys.modules['tts'] = MagicMock()
sys.modules['faster_whisper'] = MagicMock()
sys.modules['load_pdf'] = MagicMock()
sys.modules['sounddevice'] = MagicMock()

import gui

def test_ui_elements_exist():
    # check that main ui elements are created
    assert isinstance(gui.root, tk.Tk)
    assert isinstance(gui.entry, tk.Entry)
    assert isinstance(gui.btn_ask, tk.Button)
    assert isinstance(gui.btn_speech, tk.Button)
    assert isinstance(gui.txt_area, tk.scrolledtext.ScrolledText)
    assert isinstance(gui.btn_reset, tk.Button)

def test_ask_question_empty(mocker):
    # test ask question with empty input
    gui.entry.delete(0, tk.END)
    mock_messagebox = mocker.patch('gui.messagebox.showwarning')
    
    gui.ask_question()
    
    mock_messagebox.assert_called_once_with("Warning", "Please enter a question!")

def test_ask_question_with_text(mocker):
    # test ask question with valid input
    gui.entry.delete(0, tk.END)
    gui.entry.insert(0, "test query")
    
    mock_llama = mocker.patch('gui.llama_interact', return_value="mock answer")
    mock_speak = mocker.patch('gui.speak')
    
    # mock thread to run synchronously
    def fake_thread(target, daemon):
        target()
        class dummy_thread:
            def start(self): pass
        return dummy_thread()
        
    mocker.patch('gui.threading.Thread', side_effect=fake_thread)
    
    gui.ask_question()
    
    mock_llama.assert_called_once_with("test query")
    mock_speak.assert_called_once_with("mock answer")
    
    text_content = gui.txt_area.get(1.0, tk.END).strip()
    assert text_content == "mock answer"

def test_speech_to_text(mocker):
    # test speech to text functionality
    class fake_segment:
        def __init__(self, text):
            self.text = text
            
    class fake_model:
        def transcribe(self, audio, language):
            return [fake_segment("mocked speech")], None
            
    mocker.patch('gui.WhisperModel', return_value=fake_model())
    mocker.patch('gui.messagebox.showinfo')
    
    gui.speech_to_text()
    
    assert gui.entry.get() == "mocked speech"

def test_speech_to_text_empty(mocker):
    # test speech to text with no audio recognized
    class fake_model:
        def transcribe(self, audio, language):
            return [], None
            
    mocker.patch('gui.WhisperModel', return_value=fake_model())
    mocker.patch('gui.messagebox.showinfo')
    mock_warn = mocker.patch('gui.messagebox.showwarning')
    
    gui.speech_to_text()
    
    mock_warn.assert_called_once_with("No Speech", "Nothing was heard. Try again.")

def test_on_click(mocker):
    # test on click function
    def fake_thread(target, daemon):
        target()
        class dummy_thread:
            def start(self): pass
        return dummy_thread()
        
    mocker.patch('gui.threading.Thread', side_effect=fake_thread)
    mock_load = mocker.patch('gui.load_and_store_pdf')
    
    gui.on_click()
    
    mock_load.assert_called_once()

def test_reset_text_area():
    # test resetting the text area
    gui.txt_area.config(state='normal')
    gui.txt_area.delete(1.0, tk.END)
    gui.txt_area.insert(tk.END, "sample text")
    
    gui.reset_text_area()
    
    content = gui.txt_area.get(1.0, tk.END).strip()
    assert content == ""
