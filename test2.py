import numpy as np
a = 2+2

v = np.array([2,3,4])
b = np.array([3,4,5])

print(a)

print(a**2)

print(v+b)


# from help import download, load_model
import torchaudio

#Load audio signal file
file = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/p225/p225_001_mic1.wav'
print('file loaded')
signal = torchaudio.load(file)
print('done')



