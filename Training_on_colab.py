from __future__ import print_function



!pip install pydrive

'''authenticateTOGdrive required imports'''
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from google.colab import auth, drive
from oauth2client.client import GoogleCredentials
import os

'''=====================Hyperparams==========================='''


class H_params:
    '''Hyper parameters'''
    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.

    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Model
    r = 4  # Reduction factor. Do not change this.
    dropout_rate = 0.05
    e = 128  # == embedding
    d = 256  # == hidden units of Text2Mel
    c = 512  # == hidden units of SSRN
    attention_win_size = 3

    # data
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
    max_N = 180  # Maximum number of characters.
    max_T = 210  # Maximum number of mel frames.
    data = "/content/gdrive/My Drive/dt_tts/ConvDATA"
    # data = "/data/private/voice/kate"
    test_data = os.path.join(data, 'harvard_sentences.txt')

    # training scheme
    lr = 0.001  # Initial learning rate.
    BATCH_SIZE = 32  # batch size
    num_iterations = 200  # 2000000
    logdir = '/content/gdrive/My Drive/dt_tts/ConvDATA/logs'
    sampledir = 'samples'

    path_to_gdrive = '/content/gdrive'
    audio_path_file = '/content/gdrive/My Drive/dt_tts/ConvDATA/Conv.wav'
    SRT_path_file = '/content/gdrive/My Drive/dt_tts/ConvDATA/Conv.srt'
    n_lines_in_SRT = 0
    dataset_name = 'Conv'
    wavs_path = '/content/gdrive/My Drive/dt_tts/ConvDATA/wavs/'
    transcript_path = '/content/gdrive/My Drive/dt_tts/ConvDATA/'
    mel_path = '/content/gdrive/My Drive/dt_tts/ConvDATA/mel'
    mag_path = '/content/gdrive/My Drive/dt_tts/ConvDATA/mag'


def auth_mount(path_to):
    '''
    authenticates to google drive using the pydrive module
    :return: drive : a reference to the GoogleDrive() for doing any type
            of CreateFile() or GetContentFile() operations
    '''
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    '''mounting to the path'''

    drive.mount(path_to)
    print('mounted: ', path_to)
    drive_ctrl = GoogleDrive(gauth)
    print(
        '-' * 4 + 'Authnticated to GoogleDrive, Take care of adc.json file.\n it can compromise your account.' + '-' * 4)
    return drive_ctrl


'''call auth_mount in the last of your script'''
auth_mount(path_to=H_params.path_to_gdrive)

'============================================preparing_own_dataset.py==============================================================================================='
!pip install pydub tqdm librosa

from pydub import AudioSegment


def label_audio():
    i, n_sample = 1, 1
    time_step_initial_0, time_step_initial_1 = 0, 0
    label = open(os.path.join(H_params.transcript_path, 'transcript.csv'), 'w+')
    SRT_file = open(H_params.SRT_path_file, 'r')
    lines = SRT_file.readlines()
    H_params.n_lines_in_SRT = len(lines) // 4

    if not os.path.exists(H_params.wavs_path):
        if not os.path.isfile(
                os.path.join(H_params.wavs_path, 'Conv{}.wav'.format(i for i in range(H_params.n_lines_in_SRT)))):
            os.mkdir(H_params.wavs_path)

            for line in lines:
                line = line.split('\n')[0]
                if line.isnumeric():
                    continue
                if i % 3 == 0:
                    i += 1
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
                    i += 1
                else:
                    label.write(H_params.dataset_name + str(n_sample) + '|' + line + '\n')
                    split_audio(time_step_initial_0, time_step_initial_1, n_sample)
                    print(time_step_initial_0, ':', time_step_initial_1,
                          '\n' + H_params.dataset_name + str(n_sample) + '|' + line)
                    i += 1
                    n_sample += 1
            print('*************audio dataset has been created now.*****')
    else:
        print('***********audio dataset is already been available.*****')


def split_audio(time_step_initial, time_step_initial_next, N_sample):
    song = AudioSegment.from_wav(H_params.audio_path_file)
    clip = song[time_step_initial: time_step_initial_next]

    # or save to file
    clip.export(H_params.wavs_path + H_params.dataset_name + str(N_sample) + ".wav", format="wav")


label_audio()

'=====================================utils.py===================================='

import librosa
import os, copy
import matplotlib

matplotlib.use('pdf')
from scipy import signal


def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=H_params.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - H_params.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=H_params.n_fft,
                          hop_length=H_params.hop_length,
                          win_length=H_params.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(H_params.sr, H_params.n_fft, H_params.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - H_params.ref_db + H_params.max_db) / H_params.max_db, 1e-8, 1)
    mag = np.clip((mag - H_params.ref_db + H_params.max_db) / H_params.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * H_params.max_db) - H_params.max_db + H_params.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag ** H_params.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -H_params.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(H_params.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, H_params.n_fft, H_params.hop_length, win_length=H_params.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, H_params.hop_length, win_length=H_params.win_length, window="hann")


def plot_alignment(alignment, gs, dir=H_params.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')
    plt.close(fig)


def guided_attention(g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((H_params.max_N, H_params.max_T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(
                -(t_pos / float(H_params.max_T) - n_pos / float(H_params.max_N)) ** 2 / (2 * g * g))
    return W


def learning_rate_decay(init_lr, global_step, warmup_steps=4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def load_spectrograms(fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = H_params.r - (t % H_params.r) if t % H_params.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel = mel[::H_params.r, :]
    return fname, mel, mag


'=====================================data_load.py=============================================================='

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import codecs
import re
import os
import unicodedata


def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(H_params.vocab)}
    idx2char = {idx: char for idx, char in enumerate(H_params.vocab)}
    return char2idx, idx2char


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(H_params.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode == "train":
        if "Conv" in H_params.data:
            # Parse
            fpaths, text_lengths, texts = [], [], []
            max_text, min_text = 0, 20
            transcript = os.path.join(H_params.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, text = line.strip().split("|", maxsplit=2)
                if len(text) > max_text: max_text = len(text)
                if len(text) < min_text: min_text = len(text)

                fpath = os.path.join(H_params.data, "wavs", fname + ".wav")
                fpaths.append(fpath)

                text = text_normalize(text) + "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())
            text_lengths.append(min_text)
            text_lengths.append(max_text)

            return fpaths, text_lengths, texts
        else:  # nick or kate
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(H_params.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text, is_inside_quotes, duration = line.strip().split("|")
                duration = float(duration)
                if duration > 10.: continue

                fpath = os.path.join(H_params.data, fname)
                fpaths.append(fpath)

                text += "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

        return fpaths, text_lengths, texts

    else:  # synthesize on unseen test text.
        # Parse
        lines = codecs.open(H_params.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines]  # text normalization, E: EOS
        texts = np.zeros((len(sents), H_params.max_N), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts


'''all of the previous 3 methods have been subsequently called in following method'''


def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data()  # list
        minlen, maxlen = text_lengths[0], text_lengths[1]

        # Calc total batch count
        num_batch = len(fpaths) // H_params.BATCH_SIZE

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if H_params.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = os.path.join(H_params.mel_path, "{}".format(fname.replace("wav", "npy")))
                mag = os.path.join(H_params.mag_path, "{}".format(fname.replace("wav", "npy")))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath],
                                         [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, H_params.n_mels))
        mag.set_shape((None, H_params.n_fft // 2 + 1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
            input_length=text_length,
            tensors=[text, mel, mag, fname],
            batch_size=H_params.BATCH_SIZE,
            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
            num_threads=8,
            capacity=H_params.BATCH_SIZE * 4,
            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch


'''========================================================modules.py===================================================='''


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    return outputs


def normalize(inputs,
              scope="normalize",
              reuse=None):
    '''Applies layer normalization that normalizes along the last axis.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. The normalization is over the last dimension.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    outputs = tf.contrib.layers.layer_norm(inputs,
                                           begin_norm_axis=-1,
                                           scope=scope,
                                           reuse=reuse)
    return outputs


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387

    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        outputs = H * T + inputs * (1. - T)
    return outputs


'''calls normalize method'''


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           dropout_rate=0.,
           use_bias=True,
           activation_fn=None,
           training=True,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      dropout_rate: A float of [0, 1].
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
                  "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        tensor = normalize(tensor)
        if activation_fn is not None:
            tensor = activation_fn(tensor)

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor


def hc(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       dropout_rate=0.,
       use_bias=True,
       activation_fn=None,
       training=True,
       scope="hc",
       reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    _inputs = inputs
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {"inputs": inputs, "filters": 2 * filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
                  "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        H1, H2 = tf.split(tensor, 2, axis=-1)
        H1 = normalize(H1, scope="H1")
        H2 = normalize(H2, scope="H2")
        H1 = tf.nn.sigmoid(H1, "gate")
        H2 = activation_fn(H2, "info") if activation_fn is not None else H2
        tensor = H1 * H2 + (1. - H1) * _inputs

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor


def conv1d_transpose(inputs,
                     filters=None,
                     size=3,
                     dilation_rate=1,
                     stride=2,
                     padding='same',
                     dropout_rate=0.,
                     use_bias=True,
                     activation=None,
                     training=True,
                     scope="conv1d_transpose",
                     reuse=None):
    '''
        Args:
          inputs: A 3-D tensor with shape of [batch, time, depth].
          filters: An int. Number of outputs (=activation maps)
          size: An int. Filter size.
          dilation_rate: An int. Dilation rate.
          padding: Either `same` or `valid` or `causal` (case-insensitive).
          dropout_rate: A float of [0, 1].
          use_bias: A boolean.
          activation: A string.
          training: A boolean. If True, dropout is applied.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor of the shape with [batch, time*2, depth].
        '''
    with tf.variable_scope(scope, reuse=reuse):
        if filters is None:
            filters = inputs.get_shape().as_list()[-1]
        inputs = tf.expand_dims(inputs, 1)
        tensor = tf.layers.conv2d_transpose(inputs,
                                            filters=filters,
                                            kernel_size=(1, size),
                                            strides=(1, stride),
                                            padding=padding,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            use_bias=use_bias)
        tensor = tf.squeeze(tensor, 1)
        tensor = normalize(tensor)
        if activation is not None:
            tensor = activation(tensor)

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor


'''=======================================networks.py==============================='''


def TextEnc(L, training=True):
    '''
    Args:
      L: Text inputs. (B, N)

    Return:
        K: Keys. (B, N, d)
        V: Values. (B, N, d)
    '''
    i = 1
    tensor = embed(L,
                   vocab_size=len(H_params.vocab),
                   num_units=H_params.e,
                   scope="embed_{}".format(i))
    i += 1
    tensor = conv1d(tensor,
                    filters=2 * H_params.d,
                    size=1,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    i += 1

    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                        size=3,
                        rate=3 ** j,
                        dropout_rate=H_params.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i))
            i += 1
    for _ in range(2):
        tensor = hc(tensor,
                    size=3,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    activation_fn=None,
                    training=training,
                    scope="HC_{}".format(i))
        i += 1

    for _ in range(2):
        tensor = hc(tensor,
                    size=1,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    activation_fn=None,
                    training=training,
                    scope="HC_{}".format(i))
        i += 1

    K, V = tf.split(tensor, 2, -1)
    return K, V


def AudioEnc(S, training=True):
    '''
    Args:
      S: melspectrogram. (B, T/r, n_mels)

    Returns
      Q: Queries. (B, T/r, d)
    '''
    i = 1
    tensor = conv1d(S,
                    filters=H_params.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                        size=3,
                        rate=3 ** j,
                        padding="CAUSAL",
                        dropout_rate=H_params.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i))
            i += 1
    for _ in range(2):
        tensor = hc(tensor,
                    size=3,
                    rate=3,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="HC_{}".format(i))
        i += 1

    return tensor


def Attention(Q, K, V, mononotic_attention=False, prev_max_attentions=None):
    '''
    Args:
      Q: Queries. (B, T/r, d)
      K: Keys. (B, N, d)
      V: Values. (B, N, d)
      mononotic_attention: A boolean. At training, it is False.
      prev_max_attentions: (B,). At training, it is set to None.

    Returns:
      R: [Context Vectors; Q]. (B, T/r, 2d)
      alignments: (B, N, T/r)
      max_attentions: (B, T/r)
    '''
    A = tf.matmul(Q, K, transpose_b=True) * tf.rsqrt(tf.to_float(H_params.d))
    if mononotic_attention:  # for inference
        key_masks = tf.sequence_mask(prev_max_attentions, H_params.max_N)
        reverse_masks = tf.sequence_mask(H_params.max_N - H_params.attention_win_size - prev_max_attentions,
                                         H_params.max_N)[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, H_params.max_T, 1])
        paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
        A = tf.where(tf.equal(masks, False), A, paddings)
    A = tf.nn.softmax(A)  # (B, T/r, N)
    max_attentions = tf.argmax(A, -1)  # (B, T/r)
    R = tf.matmul(A, V)
    R = tf.concat((R, Q), -1)

    alignments = tf.transpose(A, [0, 2, 1])  # (B, N, T/r)

    return R, alignments, max_attentions


def AudioDec(R, training=True):
    '''
    Args:
      R: [Context Vectors; Q]. (B, T/r, 2d)

    Returns:
      Y: Melspectrogram predictions. (B, T/r, n_mels)
    '''

    i = 1
    tensor = conv1d(R,
                    filters=H_params.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    for j in range(4):
        tensor = hc(tensor,
                    size=3,
                    rate=3 ** j,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="HC_{}".format(i))
        i += 1

    for _ in range(2):
        tensor = hc(tensor,
                    size=3,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="HC_{}".format(i))
        i += 1
    for _ in range(3):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=H_params.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i))
        i += 1
    # mel_hats
    logits = conv1d(tensor,
                    filters=H_params.n_mels,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    Y = tf.nn.sigmoid(logits)  # mel_hats

    return logits, Y


def SSRN(Y, training=True):
    '''
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)

    Returns:
      Z: Spectrogram Predictions. (B, T, 1+n_fft/2)
    '''

    i = 1  # number of layers

    # -> (B, T/r, c)
    tensor = conv1d(Y,
                    filters=H_params.c,
                    size=1,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    for j in range(2):
        tensor = hc(tensor,
                    size=3,
                    rate=3 ** j,
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="HC_{}".format(i))
        i += 1
    for _ in range(2):
        # -> (B, T/2, c) -> (B, T, c)
        tensor = conv1d_transpose(tensor,
                                  scope="D_{}".format(i),
                                  dropout_rate=H_params.dropout_rate,
                                  training=training, )
        i += 1
        for j in range(2):
            tensor = hc(tensor,
                        size=3,
                        rate=3 ** j,
                        dropout_rate=H_params.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i))
            i += 1
    # -> (B, T, 2*c)
    tensor = conv1d(tensor,
                    filters=2 * H_params.c,
                    size=1,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    i += 1
    for _ in range(2):
        tensor = hc(tensor,
                    size=3,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="HC_{}".format(i))
        i += 1
    # -> (B, T, 1+n_fft/2)
    tensor = conv1d(tensor,
                    filters=1 + H_params.n_fft // 2,
                    size=1,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    i += 1

    for _ in range(2):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=H_params.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i))
        i += 1
    logits = conv1d(tensor,
                    size=1,
                    rate=1,
                    dropout_rate=H_params.dropout_rate,
                    training=training,
                    scope="C_{}".format(i))
    Z = tf.nn.sigmoid(logits)
    return logits, Z


'''================================prepro.py==========================================='''

import tqdm

# Load data
fpaths, _, _ = load_data()  # list

for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag = load_spectrograms(fpath)
    if not os.path.exists(H_params.mel_path):
        if not os.path.isfile(
                os.path.join(H_params.mel_path, 'Conv{}'.format(i for i in range(H_params.n_lines_in_SRT)))):
            os.mkdir(H_params.mel_path)
    else:
        print('------mel.npy files is already available')
    if not os.path.exists(H_params.mag_path):
        if not os.path.isfile(
                os.path.join(H_params.mag_path, 'Conv{}'.format(i for i in range(H_params.n_lines_in_SRT)))):
            os.mkdir(H_params.mag_path)
    else:
        print('-------mag.npy files are already availavble')

    np.save(os.path.join(H_params.mel_path, "{}".format(fname.replace("wav", "npy"))), mel)
    np.save(os.path.join(H_params.mag_path, "{}".format(fname.replace("wav", "npy"))), mag)

'=============================train.py========================================='
from tqdm import tqdm
import sys


class Graph:
    def __init__(self, num=1, mode="train"):
        '''
        Args:
          num: Either 1 or 2. 1 for Text2Mel 2 for SSRN.
          mode: Either "train" or "synthesize".
        '''
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set flag
        training = True if mode == "train" else False

        # Graph
        # Data Feeding
        ## L: Text. (B, N), int32
        ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
        ## mags: Magnitude. (B, T, n_fft//2+1) float32
        if mode == "train":
            self.L, self.mels, self.mags, self.fnames, self.num_batch = get_batch()
            self.prev_max_attentions = tf.ones(shape=(H_params.BATCH_SIZE,), dtype=tf.int32)
            self.gts = tf.convert_to_tensor(guided_attention())
        else:  # Synthesize
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, H_params.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

        if num == 1 or (not training):
            with tf.variable_scope("Text2Mel"):
                # Get S or decoder inputs. (B, T//r, n_mels)
                self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

                # Networks
                with tf.variable_scope("TextEnc"):
                    self.K, self.V = TextEnc(self.L, training=training)  # (N, Tx, e)

                with tf.variable_scope("AudioEnc"):
                    self.Q = AudioEnc(self.S, training=training)

                with tf.variable_scope("Attention"):
                    # R: (B, T/r, 2d)
                    # alignments: (B, N, T/r)
                    # max_attentions: (B,)
                    self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                             mononotic_attention=(not training),
                                                                             prev_max_attentions=self.prev_max_attentions)
                with tf.variable_scope("AudioDec"):
                    self.Y_logits, self.Y = AudioDec(self.R, training=training)  # (B, T/r, n_mels)
        else:  # num==2 & training. Note that during training,
            # the ground truth melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(self.mels, training=training)

        if not training:
            # During inference, the predicted melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(self.Y, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if training:
            if num == 1:  # Text2Mel
                # mel L1 loss
                self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))

                # mel binary divergence loss
                self.loss_bd1 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.mels))

                # guided_attention loss
                self.A = tf.pad(self.alignments, [(0, 0), (0, H_params.max_N), (0, H_params.max_T)], mode="CONSTANT",
                                constant_values=-1.)[:, :H_params.max_N, :H_params.max_T]
                self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
                self.loss_att = tf.reduce_sum(tf.abs(self.A * self.gts) * self.attention_masks)
                self.mask_sum = tf.reduce_sum(self.attention_masks)
                self.loss_att /= self.mask_sum

                # total loss
                self.loss = self.loss_mels + self.loss_bd1 + self.loss_att

                tf.summary.scalar('train/loss_mels', self.loss_mels)
                tf.summary.scalar('train/loss_bd1', self.loss_bd1)
                tf.summary.scalar('train/loss_att', self.loss_att)
                tf.summary.image('train/mel_gt', tf.expand_dims(tf.transpose(self.mels[:1], [0, 2, 1]), -1))
                tf.summary.image('train/mel_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
            else:  # SSRN
                # mag L1 loss
                self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))

                # mag binary divergence loss
                self.loss_bd2 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits, labels=self.mags))

                # total loss
                self.loss = self.loss_mags + self.loss_bd2

                tf.summary.scalar('train/loss_mags', self.loss_mags)
                tf.summary.scalar('train/loss_bd2', self.loss_bd2)
                tf.summary.image('train/mag_gt', tf.expand_dims(tf.transpose(self.mags[:1], [0, 2, 1]), -1))
                tf.summary.image('train/mag_hat', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))

            # Training Scheme
            self.lr = learning_rate_decay(H_params.lr, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar("lr", self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -1., 1.)
                self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # argument: 1 or 2. 1 for Text2mel, 2 for SSRN.
    num = sys.argv[1]

    g = Graph(num=num);
    print("Training Graph loaded")

    logdir = H_params.logdir + "-" + str(num)
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                    if num == 1:
                        # plot alignment
                        alignments = sess.run(g.alignments)
                        plot_alignment(alignments[0], str(gs // 1000).zfill(3) + "k", logdir)

                # break
                if gs > H_params.num_iterations: break

print("Done")

'============================synhesize.py========================================='

from tqdm import tqdm
from scipy.io.wavfile import write


def synthesize():
    # Load data
    L = load_data("synthesize")

    # Load graph
    g = Graph(mode="synthesize");
    print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(H_params.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(H_params.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), H_params.max_T, H_params.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(H_params.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(H_params.sampledir): os.makedirs(H_params.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i + 1)
            wav = spectrogram2wav(mag)
            write(H_params.sampledir + "/{}.wav".format(i + 1), H_params.sr, wav)


if __name__ == '__main__':
    synthesize()
    print("Done")
