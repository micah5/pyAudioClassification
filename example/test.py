import numpy as np
from pyaudioclassification import feature_extraction, train, predict, print_leaderboard

parent_dir = './example'

# step 1: preprocessing
if np.DataSource().exists("%s/feat.npy" % parent_dir) and np.DataSource().exists("%s/label.npy" % parent_dir):
    features, labels = np.load('%s/feat.npy' % parent_dir), np.load('%s/label.npy' % parent_dir)
else:
    features, labels = feature_extraction('%s/data/' % parent_dir)
    np.save('%s/feat.npy' % parent_dir, features)
    np.save('%s/label.npy' % parent_dir, labels)

# step 2: training
if np.DataSource().exists("%s/model.h5" % parent_dir):
    from keras.models import load_model
    model = load_model('%s/model.h5' % parent_dir)
else:
    model = train(features, labels, epochs=500)
    model.save('%s/model.h5' % parent_dir)

# step 3: prediction
pred = predict(model, '%s/test.wav' % parent_dir)
print pred
print_leaderboard(pred, '%s/data/' % parent_dir)
