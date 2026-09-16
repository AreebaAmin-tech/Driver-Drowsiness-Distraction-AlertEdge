import threading
import numpy as np
import pygame


class AlertSound:

  def __init__(self):
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    self.is_playing = False
    self._sound = self._generate_beep_sound(frequency=1000, duration_ms=400)

  def _generate_beep_sound(self, frequency=1000, duration_ms=400):
    sample_rate = 44100
    n_samples = int(round(duration_ms * 0.001 * sample_rate))
    buf = np.sin(2 * np.pi * frequency * np.arange(n_samples) / sample_rate)
    mono = (buf * 32767).astype(np.int16)

    # Duplicate mono channel into 2 channels for stereo output
    stereo = np.column_stack((mono, mono))
    return pygame.sndarray.make_sound(stereo)

  def _play_worker(self):
    self._sound.play()
    pygame.time.wait(400)
    self.is_playing = False

  def trigger(self):
    """Plays the alert tone asynchronously without blocking the video pipeline."""
    if not self.is_playing:
      self.is_playing = True
      threading.Thread(target=self._play_worker, daemon=True).start()