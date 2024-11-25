import pyaudio
import numpy as np

class SoundMonitor:
    def _init_(self, silence_threshold=1000, noise_warning_threshold=3000):
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.SILENCE_THRESHOLD = silence_threshold
        self.NOISE_WARNING_THRESHOLD = noise_warning_threshold
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

    def analyze_sound(self):
        data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.int16)
        amplitude = np.abs(data).mean()
        if amplitude < self.SILENCE_THRESHOLD:
            return "Silence maintained."
        elif amplitude > self.NOISE_WARNING_THRESHOLD:
            return "⚠ Noise detected! Please maintain silence."
        
        frequencies = np.fft.fft(data)
        frequencies = np.abs(frequencies[:self.CHUNK // 2])
        dominant_frequency = np.argmax(frequencies) * (self.RATE / self.CHUNK)

        if dominant_frequency < 300:
            return "Sound detected: Low-frequency sound (e.g., hum or rumble)."
        elif 300 <= dominant_frequency < 3000:
            return "Sound detected: Mid-frequency sound (possibly human speech)."
        else:
            return "Sound detected: High-frequency sound (e.g., sharp noise or whistle)."

    def start_monitoring(self):
        try:
            while True:
                print(self.analyze_sound())
        except KeyboardInterrupt:
            print("\nStopping monitoring.")
        finally:
            self.stop()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Audio stream closed.")