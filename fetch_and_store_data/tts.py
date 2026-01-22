from gtts import gTTS
import os
import tempfile
import platform
import subprocess
import signal

# Global variable to track current playing process
current_process = None

def stop():
    """Stops the currently playing audio."""
    global current_process
    if current_process:
        try:
            current_process.terminate()
            current_process.wait(timeout=1)
        except:
            current_process.kill()
        finally:
            current_process = None

def pause():
    """Pauses the currently playing audio (Linux only)."""
    global current_process
    if current_process and platform.system() == "Linux":
        current_process.send_signal(signal.SIGSTOP)

def resume():
    """Resumes the currently paused audio (Linux only)."""
    global current_process
    if current_process and platform.system() == "Linux":
        current_process.send_signal(signal.SIGCONT)

def speak(text: str, lang: str = "en"):
    global current_process
    
    # Singleton check: Stop ANY previous audio before starting new one
    stop()
    
    tts = gTTS(text=text, lang=lang)

    # Save to a temp
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    tmp_file.close() # Close file handle so player can open it

    if platform.system() == "Windows":
        os.startfile(tmp_file.name)
        # Windows startfile doesn't return a process we can easily control this way without comtypes
        # For this iteration, control is Linux optimized as per User OS.
    elif platform.system() == "Linux":
        # Check for players in order or availability
        player_cmd = None
        if os.system("which paplay > /dev/null") == 0:
            player_cmd = ["paplay", tmp_file.name]
        elif os.system("which mpg123 > /dev/null") == 0:
            player_cmd = ["mpg123", tmp_file.name]
        elif os.system("which aplay > /dev/null") == 0:
             # aplay needs help with mp3 usually, but let's try
             print("Trying aplay...") 
             player_cmd = ["aplay", tmp_file.name]
        
        if player_cmd:
            # Launch async process and track it
            current_process = subprocess.Popen(player_cmd)
            current_process.wait() # Wait for finish if we want logic here, but for background task we might want to wait
            # If we wait here, it blocks the background task thread, which is fine.
            # But the STOP command runs in main thread/other thread and kills controls this Popen object.
        else:
            print("Warning: No suitable audio player (paplay, mpg123) found. Please install one.")
    else:
        print("Platform not supported!")
        pass