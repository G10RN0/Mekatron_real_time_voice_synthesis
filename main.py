#importujemy
#krok 1
import pyaudio
import wave
from pydub import AudioSegment
import os
from os.path import exists, join
import sys

project_name = 'mekatron2'
sys.path.append(join(project_name, 'waveglow/'))
sys.path.append(project_name)

import gdown
from speech_to_text import *
import os
import glob

#krok 3
import IPython.display as ipd
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
from denoiser import Denoiser
import datetime
from functools import lru_cache

#koniec kroku 3

d = 'https://drive.google.com/uc?id='

graph_width = 900
graph_height = 360

thisdict = {}

hparams = None
model = None
waveglow = None
denoiser = None

tacotron2_pretrained_model = 'TTS'
waveglow_pretrained_model = 'waveglow.pt'


#krok2
#pobieramy modele tak i tak

@lru_cache(maxsize=None)
def ARPA(text):
    global thisdict
    out = ''
    for word_ in text.split(" "):
        word=word_; end_chars = ''
        while any(elem in word for elem in r"!?,.;") and len(word) > 1:
            if word[-1] == '!': end_chars = '!' + end_chars; word = word[:-1]
            if word[-1] == '?': end_chars = '?' + end_chars; word = word[:-1]
            if word[-1] == ',': end_chars = ',' + end_chars; word = word[:-1]
            if word[-1] == '.': end_chars = '.' + end_chars; word = word[:-1]
            if word[-1] == ';': end_chars = ';' + end_chars; word = word[:-1]
            else: break
        try: word_arpa = thisdict[word.upper()]
        except: word_arpa = ''
        if len(word_arpa)!=0: word = "{" + str(word_arpa) + "}"
        out = (out + " " + word + end_chars).strip()
    if out[-1] != ";": out = out + ";"
    return out

def setup():
    global thisdict
    global hparams
    global model
    global waveglow
    global denoiser
    d = 'https://drive.google.com/uc?id='

    force_download_TT2 = False
    force_download_waveglow = False

    #link do modelu tacotron2 jeśli jeszcze niemamy
    TT2_link = "1df5KQ0WwxllwLsEbp7vdzuqHshhuM5JU"

    #link do modelu waveglow jeśli jeszcze niemamy
    waveglow_link = "17xuBnKr6gtGfR21Hmsgx_rk8kKPrEWrn"

    if not exists(tacotron2_pretrained_model) or force_download_TT2:
        print("Pobieranie modelu Tacotron2")             
        gdown.download(d+TT2_link.strip(), tacotron2_pretrained_model, quiet=False) 
        print("Model Tacotron2 pobrany pomyślnie")

    if not exists(waveglow_pretrained_model) or force_download_waveglow:
        print("Pobieranie modelu Waveglow")
        gdown.download(d+waveglow_link.strip(), waveglow_pretrained_model, quiet=False)
        print("Model Waveglow pobrany pomyślnie")
    
    for line in reversed((open('merged.dict.txt', "r", encoding="utf8").read()).splitlines()):
        thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()

    #torch.set_grad_enabled(False)

    # initialize Tacotron2 with the pretrained model
    hparams = create_hparams()

    #gdown.download(d+r'1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW', 'pretrained', quiet=False); print("Model Downloaded")
    #gdown.download(d+r'1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e', 'config.json', quiet=False); print("Config Downloaded")
    try:
        os.mkdir("test_files")
    except FileExistsError:
        pass

    #częstotliwość próbkowania**. Większa częstotliwość próbkowania zwiększa prędkość głosu i jego ton. (domyślnie jest 22050). Lepiej nie zmieniać bazowej wartości. 
    hparams.sampling_rate = 22050 #@param{type:'number', min:'0', max:'384000'}
    #Maksymalna ilość iteracji dekodera** wpływa na długość wygenerowanego głosu. 

    #Można zwiększyć, by syntezator mógł wygenerować dłuższy głos (domyślnie 1000)
    hparams.max_decoder_steps = 2000
    hparams.gate_threshold = 0.1 # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
    model = Tacotron2(hparams)
    model.load_state_dict(torch.load(tacotron2_pretrained_model)['state_dict'])
    _ = model.cuda().eval().half()

    # Załaduj Waveglow
    waveglow = torch.load(waveglow_pretrained_model)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

def synteza(text):
    global model
    global waveglow
    global denoiser

    denoise_strength =  0.001

    equalize = True
    gan = False

    _sigma = 1
    speed_multiplier = 1
    raw_input = False 

    files = glob.glob('test_files/*')
    for f in files:
        os.remove(f)

    for i in text.split("\n"):
        if len(i) < 1: 
            continue
        print(i)
        if raw_input:
            if i[-1] != ";": 
                i=i+";" 
        else: 
            i = ARPA(i)
        print(i)
        
        sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=_sigma)
        audio_denoised = denoiser(audio, strength=denoise_strength)[:, 0]

        print("Zwykła wersja:")

        audio = ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate * speed_multiplier)
        audio = AudioSegment(audio.data, frame_rate=hparams.sampling_rate * speed_multiplier, sample_width=2, channels=1)
        audio.export("test_files/testnt.wav", format="wav", bitrate="32k")
            
        os.system('ffmpeg -loglevel quiet -i "test_files/testnt.wav" -ss 0.0020 -vcodec copy -acodec copy "test_files/test.wav"') 
            
        if equalize:
            
            os.system('ffmpeg -loglevel quiet -y -i "test_files/test.wav" -ac 2 -af "aresample=44100:resampler=soxr:precision=15, equalizer=f=50:width_type=o:width=0.75:g=3.6, equalizer=f=3000:width_type=o:width=1.0:g=2.0, equalizer=f=10000:width_type=o:width=1.0:g=4.0" "test_EQ.wav"')

        if gan:
            if equalize:
                print("Gan z EQ:")
                print("")
                print("---------Logi generacji GAN---------")
                print("")
                os.system('python hifi-gan/inference.py --checkpoint_file pretrained') 
                print("")
                print("---------Koniec logów generacji GAN---------")
                print("")
                os.system('ffmpeg -loglevel quiet -y -i "generated_files/test_generated.wav" -ac 2 -af "aresample=44100:resampler=soxr:precision=15, equalizer=f=50:width_type=o:width=0.75:g=3.6, equalizer=f=3000:width_type=o:width=1.0:g=2.0, equalizer=f=10000:width_type=o:width=1.0:g=4.0" "generated_files/test_EQ.wav"')
            else:
                print("Gan z EQ:")
                print("")
                print("---------Logi generacji GAN---------")
                print("")
                os.system('python hifi-gan/inference.py --checkpoint_file pretrained') 
                print("")
                print("---------Koniec logów generacji GAN---------")
                print("")
            
    print("DONE")
    
    

def play(path = 'test_EQ.wav'):
  #import sounddevice as sd

  # Create an interface to PortAudio
  p = pyaudio.PyAudio()

  #print(sd.query_devices())

  # Set chunk size of 1024 samples per data frame
  chunk = 1024 

  # Open the sound file 
  wf = wave.open(path, 'rb')

  # Open a .Stream object to write the WAV file to
  # 'output = True' indicates that the sound will be played rather than recorded
  stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                  channels = wf.getnchannels(),
                  rate = wf.getframerate(),
                  output = True,
                  output_device_index = 10)

  # Read data in chunks
  data = wf.readframes(chunk)

  # Play the sound by writing the audio data to the stream
  while data != b'':
      stream.write(data)
      data = wf.readframes(chunk)

  # Close and terminate the stream
  stream.close()
  p.terminate()

if __name__ == '__main__':
        
    #kompilujemy
    setup()

    #główny kod
    print('już')

    while True:
        text = recognize_speech()
        if text != None:
            begin_time = datetime.datetime.now()
            synteza(text)
            print(datetime.datetime.now() - begin_time)
            play()

#python text_to_speech.py