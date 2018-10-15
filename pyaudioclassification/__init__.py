import os
from feat_extract import parse_audio_files
import numpy as np
import models
from keras.optimizers import Adam
#from models import svm, nn, cnn

def feature_extraction(data_path, save=False, file_names=('feat', 'label')):
    r = os.listdir(data_path)
    r.sort()
    features, labels = parse_audio_files(data_path, r)
    if save == True:
        np.save('%s.npy' % file_names[0], features)
        np.save('%s.npy' % file_names[1], labels)
    return features, labels

def save_features(features, labels, file_names=('feat', 'label')):
    np.save('%s.npy' % file_names[0], features)
    np.save('%s.npy' % file_names[1], labels)

def load_features(feat_path='feat.npy', label_path="label.npy"):
    features, labels = np.load(feat_path), \
        np.load(label_path).ravel()
    return features, labels

def train(features, labels, type='cnn', num_classes=None, print_summary=False, test_split=0, save_model=False, lr=0.005, loss_type=None, epochs):
    from sklearn.model_selection import train_test_split
    if num_classes == None: num_classes = np.max(labels, axis=0)
    if test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=233)

    model = getattr(models, type)(num_classes)
    if print_summary == True: model.summary()

    if loss_type == None:
        loss_type = 'binary' if num_classes > 0 else 'categorical'
    model.compile(optimizer=Adam(lr=lr),
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

def save_model(model, name):
    model.save('%s.h5' % name)

def load_model(path):
    return load_model(path)

def predict(data_path, model):
    x_data = parse_audio_file(data_path)
    X_train = np.expand_dims(x_data, axis=2)
    pred = model.predict(X_train)
    return pred

def print_leaderboard(pred):
    sorted = np.argsort(pred)
    count = 0
    for index in (-pred).argsort()[0]:
        print count, '.', r[index + 1], '(index %s)' % index
        count += 1
