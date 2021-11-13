import sys, os
import librosa
import numpy as np
import glob
from pydub import AudioSegment, effects

def sound_normalize(wav_path, output_dir):
    path = os.path.basename(wav_path)
    song = AudioSegment.from_wav(wav_path)
    song = effects.normalize(song)
    #song = song + 10
    song.export(os.path.join(output_dir, path), "wav")


if __name__=='__main__':
    wav_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    flist = [file for file in glob.glob(os.path.join(wav_dir, '*.wav'))]
    flist.sort()
    for wav_file in flist:
        sound_normalize(wav_file, output_dir)
