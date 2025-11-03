from gtts import gTTS
import os
import tempfile
import platform



def speak(text: str, lang: str = "en"):
    tts = gTTS(text=text, lang=lang)

    # Save to a temp
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)


    if platform.system() == "Windows":
        os.startfile(tmp_file.name)
    else:
        exit()#TODO implement other OS