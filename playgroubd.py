from utils.pydub.pydub import audio_segment, effects, scipy_effects, silence
from utils.pydub.pydub import AudioSegment
import numpy as np

sound = AudioSegment.from_wav('samples\\Ses01F_impro01_F002_neu.wav')
ch = sound.channels
fr = sound.