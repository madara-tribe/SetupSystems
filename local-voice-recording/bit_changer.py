import sys
import glob
import soundfile
import librosa
from scipy.io import wavfile # read write
import os

def check_wav_bit(file):
    print(file)
    fs, data = wavfile.read(file)
    print(data.dtype, 'bit')

def convert_wav_bit(file, save_path, bit=16):
    d, s = librosa.load(file, sr=16000)
    print(d.max(), d.min())
    if bit==16:
        soundfile.write(save_path, d, s, subtype='PCM_16')
    elif bit==32:
        soundfile.write(save_path, d, s, subtype='FLOAT')
    elif bit==24:
        soundfile.write(save_path, d, s, subtype='PCM_24')
    elif bit==8:
        soundfile.write(save_path, d, s, subtype='PCM_8')
        
if __name__=='__main__':
    print('python ~.py input_dir target_dir Nbit')
    from_dir = sys.argv[1]
    save_dir = sys.argv[2]
    BIT = int(sys.argv[3])
    flist = [file for file in glob.glob(os.path.join(from_dir, '*.wav'))]
    flist.sort()
    for wav_file in flist:
        #check_wav_bit(wav_file)
        path = os.path.basename(wav_file)
        convert_wav_bit(wav_file, save_path=os.path.join(save_dir, path), bit=BIT)
