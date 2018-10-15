import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
from tqdm import tqdm

def extract_feature(file_name):
    """Generates feature input (mfccs, chroma, mel, contrast, tonnetz).
    -*- author: mtobeiyf https://github.com/mtobeiyf/audio-classification -*-
    """
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def parse_audio_files(parent_dir, sub_dirs, file_ext=None, verbose=True):
    """Parses directory in search of specified file types, then compiles feature data from them.
    -*- adapted from code by mtobeiyf https://github.com/mtobeiyf/audio-classification -*-
    """
    # by default test for only these types
    if file_ext == None:
        file_types = ['*.ogg', '*.wav']
    else:
        file_types = []
        file_types.push(file_ext)
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for file_ext in file_types:
            # file names
            iter = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))
            if len(iter) > 0:
                if verbose: print 'Reading', os.path.join(parent_dir, sub_dir, file_ext), '...'
                for fn in tqdm(iter):
                    ext_features = get_ext_features(fn)
                    features = np.vstack([features, ext_features])
                    labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype = np.int)

def get_ext_features(fn):
    """Returns features for individual audio file.
    -*- adapted from code by mtobeiyf https://github.com/mtobeiyf/audio-classification -*-
    """
    try:
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
    except Exception as e:
        print("[Error] extract feature error. %s" % (e))
        continue
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return ext_features

def parse_audio_file(fn):
    """Returns features of single audio file
    -*- adapted from code by mtobeiyf https://github.com/mtobeiyf/audio-classification -*-
    """
    features = np.empty((0,193))
    ext_features = get_ext_features(fn)
    features = np.vstack([features,ext_features])
    return np.array(features)
