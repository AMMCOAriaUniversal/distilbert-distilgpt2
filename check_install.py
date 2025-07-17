# check_install.py
try:
    import openai_whisper
    print("openai-whisper is installed.")
except ImportError:
    print("Failed to import openai-whisper.")
