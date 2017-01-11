import os
import numpy as np
import scipy.io.wavfile
filename = "ZarathustraSmall.txt"
text = open(filename).read()
text2 = text.lower()
text2

### TEXT-TO-SPEECH: SPEAKS THE DOCUMENT

os.system('say "{}"'.format(text2))

### EXPORTS THE SPOKEN TEXT AS WAV FILE

from gtts import gTTS
wav = gTTS(text=text2, lang='en')
wav.save("Zarathustra.wav")
