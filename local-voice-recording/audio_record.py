import sys
import pyaudio  
import wave 

def audio_device_id():
    iAudio = pyaudio.PyAudio()
    for x in range(0, iAudio.get_device_count()): 
        print(iAudio.get_device_info_by_index(x))

iDeviceIndex = 0 #録音デバイスのインデックス番号

def record_audio(wav_file_name="sample1.wav"):
    RECORD_SECONDS = 5 #録音する時間の長さ（秒）
    WAVE_OUTPUT_FILENAME = wav_file_name

    #基本情報の設定
    FORMAT = pyaudio.paInt16 #音声のフォーマット
    CHANNELS = 1             #モノラル
    RATE = 44100             #サンプルレート
    CHUNK = 2**11            #データ点数
    audio = pyaudio.PyAudio() #pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            input_device_index = iDeviceIndex, #録音デバイスのインデックス番号
            frames_per_buffer=CHUNK)

    #--------------録音開始---------------

    print ("recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)


    print ("finished recording")

    #--------------録音終了---------------

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    
if __name__=='__main__':
    id_show=None
    if id_show is True:
        audio_device_id()
    wav_filename = str(sys.argv[1])
    record_audio(wav_file_name=wav_filename)
