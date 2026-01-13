import pyttsx3
import time

class AudioManager:
    def __init__(self, rate=160, gap=2.0):
        self.engine = pyttsx3.init(driverName="sapi5")
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", 1.0)

        voices = self.engine.getProperty("voices")
        self.engine.setProperty("voice", voices[0].id)  # stable English voice

        self.last_time = 0
        self.speak_gap = gap

    def speak(self, text):
        now = time.time()
        if now - self.last_time >= self.speak_gap:
            try:
                self.engine.stop()
                self.engine.say(text)
                self.engine.runAndWait()
                self.last_time = now
            except RuntimeError:
                pass
