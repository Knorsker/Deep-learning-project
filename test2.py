import numpy as np
import matplotlib.pyplot as plt

a = 2+2

v = np.array([2,3,4])
b = np.array([3,4,5])

print(a)

print(a**2)

print(v+b)

plt.bar(v,b)
plt.show()


# import dac
# from audiotools import AudioSignal

# Download a model
# model_path = dac.utils.download(model_type="44khz")
# model = dac.DAC.load(model_path)

# model.to('cuda')

# Load audio signal file
# file = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/p225/p225_001_mic1.wav'
# signal = AudioSignal(file)
# print(signal)



