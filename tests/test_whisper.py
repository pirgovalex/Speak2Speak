import sys
import pytest
from unittest.mock import MagicMock

# mock modules to avoid loading real llm or gui
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

def sync_thread(target, daemon=False):
    # fake thread that just runs target synchronously when start is called
    class FakeThread:
        def start(self):
            target()
    return FakeThread()

def fake_after(delay, func, *args):
    # fake after that calls the function immediately
    func(*args)

def test_speech_to_text_success(mocker):
    # mock threading to run synchronously
    mocker.patch("gui.threading.Thread", side_effect=sync_thread)
    
    # mock gui.entry.after to run synchronously
    mocker.patch.object(gui.entry, "after", side_effect=fake_after)

    # mock sounddevice
    mock_sd = sys.modules["sounddevice"]
    mocker.patch.object(mock_sd, "rec", return_value="fake_audio")
    mocker.patch.object(mock_sd, "wait")

    # mock whisper model
    mock_model_instance = MagicMock()
    mocker.patch.object(gui, "WhisperModel", return_value=mock_model_instance)

    # setup fake segments
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_model_instance.transcribe.return_value = ([mock_segment], None)

    # reset gui mocks
    gui.entry.delete.reset_mock()
    gui.entry.insert.reset_mock()
    gui.messagebox.showinfo.reset_mock()

    # run target function
    gui.speech_to_text()

    # verify function calls
    gui.messagebox.showinfo.assert_called_once_with("Recording", "Speak now... (6 seconds)")
    mock_sd.rec.assert_called_once()
    mock_sd.wait.assert_called_once()
    gui.WhisperModel.assert_called_once_with("tiny.en")
    mock_model_instance.transcribe.assert_called_once_with("fake_audio", language="en")

    gui.entry.delete.assert_called_once_with(0, "end")
    gui.entry.insert.assert_called_once_with(0, "hello world")

def test_speech_to_text_no_speech(mocker):
    # mock threading to run synchronously
    mocker.patch("gui.threading.Thread", side_effect=sync_thread)

    # mock gui.entry.after to run synchronously
    mocker.patch.object(gui.entry, "after", side_effect=fake_after)

    # mock sounddevice
    mock_sd = sys.modules["sounddevice"]
    mocker.patch.object(mock_sd, "rec", return_value="fake_audio")
    mocker.patch.object(mock_sd, "wait")

    # mock whisper model
    mock_model_instance = MagicMock()
    mocker.patch.object(gui, "WhisperModel", return_value=mock_model_instance)

    # setup empty segments
    mock_model_instance.transcribe.return_value = ([], None)

    # reset gui mocks
    gui.entry.insert.reset_mock()
    gui.messagebox.showinfo.reset_mock()
    gui.messagebox.showwarning.reset_mock()

    # run target function
    gui.speech_to_text()

    # verify function calls
    gui.messagebox.showinfo.assert_called_once_with("Recording", "Speak now... (6 seconds)")
    gui.WhisperModel.assert_called_once_with("tiny.en")
    mock_model_instance.transcribe.assert_called_once_with("fake_audio", language="en")

    # verify no insert but warning is shown
    gui.entry.insert.assert_not_called()
    gui.messagebox.showwarning.assert_called_once_with("No Speech", "Nothing was heard. Try again.")
