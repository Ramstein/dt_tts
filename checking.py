audio_path_file = r'C:\Users\Ramstein\Music\Conv.wav'
SRT_path_file = r'C:\Users\Ramstein\\Downloads\Conv.srt'
dataset_name = 'Conv'


SRT_file = open(SRT_path_file, 'r')
lines = SRT_file.readlines()
lines_size = len(lines)
print(lines_size)
print('n_samples: ', lines_size//4)
for line in lines:
    line = line.split('\n')[0]
    # if line.isspace():
    #     continue
    if line.isnumeric():
        continue

    print(line)