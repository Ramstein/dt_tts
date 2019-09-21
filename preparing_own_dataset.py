from pydub import AudioSegment
import os


audio_path_file = r'C:\Users\Ramstein\Music\Conv.wav'
SRT_path_file = r'C:\Users\Ramstein\\Downloads\Conv.srt'
dataset_name = 'Conv'


def label_audio():
    i, n_sample = 1, 1
    time_step_initial_0 , time_step_initial_1= 0, 0
    label = open('transcript.csv', 'w+')
    SRT_file = open(SRT_path_file, 'r')
    lines = SRT_file.readlines()

    for line in lines:
        line = line.split('\n')[0]
        if line.isnumeric():
            continue
        if i % 3==0:
            i+=1
            continue
        if '-->' in line:
            time_step = line.split(' --> ', maxsplit=2)
            time_step_initial_0 = int(
                    time_step[0].split(',')[0].split(':', maxsplit=3)[-3]) * 3600 * 1000 + int(
                    time_step[0].split(',')[0].split(':', maxsplit=3)[-2]) * 60 * 1000 + int(
                    time_step[0].split(',')[0].split(':', maxsplit=3)[-1]) * 1000 + int(time_step[0].split(',')[-1])
            time_step_initial_1 = int(
                time_step[-1].split(',')[0].split(':', maxsplit=3)[-3]) * 3600 * 1000 + int(
                time_step[-1].split(',')[0].split(':', maxsplit=3)[-2]) * 60 * 1000 + int(
                time_step[-1].split(',')[0].split(':', maxsplit=3)[-1]) * 1000 + int(
                time_step[-1].split(',')[-1])

            i+=1
        else:
            label.write(dataset_name+str(n_sample)+'|'+line +'\n')
            split_audio(time_step_initial_0, time_step_initial_1, n_sample)
            print(time_step_initial_0, ':', time_step_initial_1,
                  '\n' + dataset_name + str(n_sample) + '|' + line)
            i += 1
            n_sample += 1


def split_audio(time_step_initial, time_step_initial_next, N_sample):
    song = AudioSegment.from_wav(audio_path_file)
    eight_sec_slice = song[time_step_initial: time_step_initial_next]

    if not os.path.exists('/wav'):
        os.mkdir('/wav')
    # or save to file
    eight_sec_slice.export('/wav/'+dataset_name + str(N_sample) + ".wav", format="wav")


label_audio()
