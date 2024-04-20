import numpy as np
from pedalboard import Pedalboard, Chorus, Compressor, Delay, Distortion, Flanger, Phaser, Reverb, Tremolo, Wah
from pedalboard.guitar import GuitarDistortion, GuitarReverb, GuitarDelay, GuitarChorus, GuitarFlanger, GuitarPhaser, GuitarTremolo, GuitarWah
from pedalboard.utils import get_effect_names
from pedalboard.io import AudioFileSink, AudioFileSource
import random


def apply_compression(audio, sr, threshold=-20.0, ratio=4.0, attack=10, release=100):
    compressor = Compressor(threshold_db=threshold, ratio=ratio, attack_ms=attack, release_ms=release)
    with Pedalboard([compressor], sr) as board:
        audio = board(audio)
    return audio


def apply_reverb(audio, sr, room_size=0.5):
    reverb = Reverb(room_size=room_size)
    return reverb(audio, sr)


def apply_delay(audio, sr, delay_time=0.5, feedback=0.5):
    delay = Delay(delay_time_seconds=delay_time, feedback_percent=feedback)
    return delay(audio, sr)


def apply_distortion(audio, sr, drive=0.5):
    distortion = Distortion(drive_db=drive)
    return distortion(audio, sr)


def apply_chorus(audio, sr, rate=2.0, depth=0.5):
    chorus = Chorus(rate_hz=rate, depth=depth)
    return chorus(audio, sr)


def apply_phaser(audio, sr, rate=0.5, depth=1):
    phaser = Phaser(rate_hz=rate, depth=depth)
    return phaser(audio, sr)


def apply_flanger(audio, sr, rate=0.1, depth=1):
    flanger = Flanger(rate_hz=rate, depth=depth)
    return flanger(audio, sr)


def apply_tremolo(audio, sr, rate=5, depth=0.5):
    tremolo = Tremolo(rate_hz=rate, depth=depth)
    return tremolo(audio, sr)


# Random effect applier
def apply_random_effects(audio, sr):
    effects_list = [
        apply_reverb, apply_delay, apply_distortion,
        apply_chorus, apply_phaser, apply_flanger, apply_tremolo
    ]

    # Choose a random number of effects to apply
    num_effects = random.randint(1, len(effects_list))
    effects_to_apply = random.sample(effects_list, num_effects)

    # Apply the effects
    for effect in effects_to_apply:
        # Generate random parameters for the effects
        if effect == apply_reverb:
            audio = effect(audio, sr, room_size=random.uniform(0.1, 1))
        elif effect == apply_delay:
            audio = effect(audio, sr, delay_time=random.uniform(0.1, 1), feedback=random.uniform(0, 0.75))
        elif effect == apply_distortion:
            audio = effect(audio, sr, drive=random.uniform(5, 20))
        elif effect == apply_chorus:
            audio = effect(audio, sr, rate=random.uniform(0.5, 4), depth=random.uniform(0.1, 0.7))
        elif effect == apply_phaser:
            audio = effect(audio, sr, rate=random.uniform(0.1, 5), depth=random.uniform(0.1, 1))
        elif effect == apply_flanger:
            audio = effect(audio, sr, rate=random.uniform(0.05, 0.2), depth=random.uniform(0.5, 1))
        elif effect == apply_tremolo:
            audio = effect(audio, sr, rate=random.uniform(1, 8), depth=random.uniform(0.1, 1))

    return audio



