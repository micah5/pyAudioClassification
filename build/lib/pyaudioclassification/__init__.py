import os
from feat_extract import parse_audio_files
import numpy as np
import models
from keras.optimizers import Adam
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
    if num_classes == None: num_classes = np.max(labels, axis=0)
    if test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=233)

    model = getattr(models, type)(num_classes)
    if print_summary == True: model.summary()

    if loss_type == None:
        loss_type = 'binary' if num_classes > 0 else 'categorical'
    model.compile(optimizer=SGD(lr=lr),
                  loss='%s_crossentropy' % loss_type,
                  metrics=['accuracy'])

    if loss_type == 'categorical':
        y_train = keras.utils.to_categorical(y_train - 1, num_classes=num_classes)
        if test_split > 0: y_test = keras.utils.to_categorical(y_test - 1, num_classes=num_classes)
    else:
        y_train = y_train - 1
        if test_split > 0: y_test = y_test - 1

    X_train = np.expand_dims(X_train, axis=2)
    if test_split > 0:
        X_test = np.expand_dims(X_test, axis=2)

    model.fit(X_train, y_train, batch_size=64, epochs=epochs)
    if save_model == True: model.save('my_model.h5')
    if test_split > 0:
        score, acc = model.evaluate(X_test, y_test, batch_size=16)
        print('Test score:', score)
        print('Test accuracy:', acc)

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

def print_leaderboard(pred):
    """Pretty prints leaderboard of top matches
    """
    sorted = np.argsort(pred)
    count = 0
    for index in (-pred).argsort()[0]:
        print '%d.' % (count + 1), r[index + 1], '(index %s)' % index
        count += 1
