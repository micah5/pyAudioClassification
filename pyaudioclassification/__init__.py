import os
from feat_extract import parse_audio_files, parse_audio_file
import numpy as np
import models
from keras.utils import to_categorical
from keras.optimizers import SGD
#from models import svm, nn, cnn

def feature_extraction(data_path):
    """Parses audio files in supplied data path.
    -*- author: mtobeiyf https://github.com/mtobeiyf/audio-classification -*-
    """
    r = os.listdir(data_path)
    r.sort()
    features, labels = parse_audio_files(data_path, r)
    return features, labels

def train(features, labels, type='cnn', num_classes=None, print_summary=False,
    save_model=False, lr=0.01, loss_type=None, epochs=50, optimizer='SGD'):
    """Trains model based on provided feature & target data
    Options:
    - epochs: The number of iterations. Default is 50.
    - lr: Learning rate. Increase to speed up training time, decrease to get more accurate results (if your loss is 'jumping'). Default is 0.01.
    - optimiser: Default is 'SGD'.
    - print_summary: Prints a summary of the model you'll be training. Default is False.
    - loss_type: Classification type. Default is categorical for >2 classes, and binary otherwise.
    """
    labels = labels.ravel()
    if num_classes == None: num_classes = np.max(labels, axis=0)

    model = getattr(models, type)(num_classes)
    if print_summary == True: model.summary()

    if loss_type == None:
        loss_type = 'binary' if num_classes <= 2 else 'categorical'
    model.compile(optimizer=SGD(lr=lr),
                  loss='%s_crossentropy' % loss_type,
                  metrics=['accuracy'])

    if loss_type == 'categorical':
        y = to_categorical(labels - 1, num_classes=num_classes)
    else:
        y = labels - 1

    x = np.expand_dims(features, axis=2)

    model.fit(x, y, batch_size=64, epochs=epochs)

    return model

def predict(model, data_path):
    """Trains model based on provided feature & target data
    Options:
    - epochs: The number of iterations. Default is 50.
    - lr: Learning rate. Increase to speed up training time, decrease to get more accurate results (if your loss is 'jumping'). Default is 0.01.
    - optimiser: Default is 'SGD'.
    - print_summary: Prints a summary of the model you'll be training. Default is False.
    - type: Classification type. Default is categorical for >2 classes, and binary otherwise.
    """
    x_data = parse_audio_file(data_path)
    X_train = np.expand_dims(x_data, axis=2)
    pred = model.predict(X_train)
    return pred

def print_leaderboard(pred, data_path):
    """Pretty prints leaderboard of top matches
    """
    r = os.listdir(data_path)
    r.sort()
    sorted = np.argsort(pred)
    count = 0
    for index in (-pred).argsort()[0]:
        print '%d.' % (count + 1), r[index + 1], str(round(pred[0][index]*100)) + '%', '(index %s)' % index
        count += 1
