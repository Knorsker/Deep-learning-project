import numpy as np
a = 2+2

v = np.array([2,3,4])
b = np.array([3,4,5])

print(a)

print(a**2)

print(v+b)


# import dac
# from help import download, load_model
# import torchaudio

# # Download a model
# model_path = download(model_type="44khz")
# model = load_model(model_path)

# model.to('cuda')

# #Load audio signal file
# file = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/p225/p225_001_mic1.wav'
# signal = torchaudio.load(file)
# print(signal)

import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: test2.py p225_001_mic1.wav")
        sys.exit(1)

    input_data_file = sys.argv[1]

    # Now you can use 'input_data_file' in your script to access the data
    try:
        with open(input_data_file, 'r') as file:
            # Process the data or perform other operations
            data = file.read()
            print(f"Successfully read data from {input_data_file}:\n{data}")
    except FileNotFoundError:
        print(f"Error: File not found - {input_data_file}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

