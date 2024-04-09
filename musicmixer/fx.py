import numpy as np
from pedalboard import Pedalboard, Chorus, Compressor, Delay, Distortion, Flanger, Phaser, Reverb, Tremolo, Wah
from pedalboard.guitar import GuitarDistortion, GuitarReverb, GuitarDelay, GuitarChorus, GuitarFlanger, GuitarPhaser, GuitarTremolo, GuitarWah
from pedalboard.utils import get_effect_names
from pedalboard.io import AudioFileSink, AudioFileSource


def apply_compression(audio, sr, threshold=-20.0, ratio=4.0, attack=10, release=100):
    compressor = Compressor(threshold_db=threshold, ratio=ratio, attack_ms=attack, release_ms=release)
    with Pedalboard([compressor], sr) as board:
        audio = board(audio)
    return audio



