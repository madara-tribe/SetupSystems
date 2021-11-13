import sys, os
import librosa
import numpy as np
import soundfile
import glob

def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))
    
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")

def shift_sound(x, rate=0.99):
    return np.roll(x, int(len(x)//rate))


def convert_wav_bit(file, save_path, types=None):
    x, s = librosa.load(file, sr=16000)
    aug_x = get_aug_x(x, types=types)
    
    path = os.path.basename(wav_file)
    os.makedirs(save_path+types, exist_ok=True)
    save_file_name = os.path.join(save_path+types, types+'_'+path)
    soundfile.write(save_file_name, aug_x, s, 'PCM_24') # PCM_16' PCM_8'
        
def get_aug_x(x, types=None):
    assert types in ['noise', 'stretch'], '"noise", "stretch"'
    if types=='noise':
        aug_x = add_white_noise(x, rate=0.002)
    elif types=='stretch':
        aug_x = stretch_sound(x, rate=1.1)
    return aug_x

if __name__=='__main__':
    types='noise'
    wav_dir = sys.argv[1]
    flist = [file for file in glob.glob(os.path.join(wav_dir, '*.wav'))]
    flist.sort()
    save_dir=wav_dir
    for wav_file in flist:
        convert_wav_bit(wav_file, save_path=save_dir, types=types)
        
