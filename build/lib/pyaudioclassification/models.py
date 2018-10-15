import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def svm(num_classes):
    """Support vector machine.
    -*- ref: mtobeiyf https://github.com/mtobeiyf/audio-classification -*-
    """
    from sklearn.svm import SVC

    return SVC(C=20.0, gamma=0.00001)

def nn(num_classes):
    """Multi-layer perceptron.
    """
    model = Sequential()
    model.add(Dense(512, input_dim=193))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def cnn(num_classes):
    """1D Convolutional Neural Network.
    """
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    model = Sequential()
    model.add(Conv1D(64, 3, input_shape=(193, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 3))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
