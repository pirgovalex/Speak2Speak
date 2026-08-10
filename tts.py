from gtts import gTTS
import os
import tempfile
import platform
import subprocess


def speak(text: str, lang: str = "en"):
    tts = gTTS(text=text, lang=lang)

    # Save to a temp
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    tmp_file.close()

    if platform.system() == "Windows":
        os.startfile(tmp_file.name)
    elif platform.system() == "Linux":
        player_cmd = None
        if os.system("which mpg123 > /dev/null 2>&1") == 0:
            player_cmd = ["mpg123", tmp_file.name]
        elif os.system("which paplay > /dev/null 2>&1") == 0:
            player_cmd = ["paplay", tmp_file.name]
        elif os.system("which aplay > /dev/null 2>&1") == 0:
            player_cmd = ["aplay", tmp_file.name]

        if player_cmd:
            subprocess.run(player_cmd, check=False)
        else:
            print("Warning: no audio player found (mpg123, paplay, aplay). Install one.")
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported for TTS")