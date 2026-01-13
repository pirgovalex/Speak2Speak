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
    elif platform.system() == "Linux":
        # Check for players in order or availability
        if os.system("which paplay > /dev/null") == 0:
            os.system(f"paplay {tmp_file.name}")
        elif os.system("which mpg123 > /dev/null") == 0:
            os.system(f"mpg123 {tmp_file.name}")
        elif os.system("which aplay > /dev/null") == 0:
            # aplay often fails on mp3 but might work if gTTS output format is supported or user has plugins
            print("Trying aplay...") 
            os.system(f"aplay {tmp_file.name}")
        else:
            print("Warning: No suitable audio player (paplay, mpg123) found. Please install one.")
    else:
        print("Platform not supported!")
        exit()#not supported