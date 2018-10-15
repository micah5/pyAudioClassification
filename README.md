# pyAudioClassification
Dead simple audio classification

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg?style=flat-square)
![PyPI](https://img.shields.io/pypi/v/nine.svg?style=flat-square)

## Who is this for? 👩‍💻 👨‍💻
People who just want to classify some audio quickly, without having to dive into the world of audio analysis.
If you need something a little more involved, check out [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) or [panotti](https://github.com/drscotthawley/panotti)

## Quick install
```
pip install pyaudioclassification
```

### Requirements
* Keras
* Tensorflow
* librosa
* NumPy
* Soundfile
* tqdm

## Quick start
```python
from pyaudioclassification import feature_extraction, train, predict
features, labels = feature_extraction(<data_path>)
model = train(features, labels)
pred = predict(model, <data_path>)
```

Or, if you're feeling reckless, you could just string them together like so:
```python
pred = predict(train(feature_extraction(<training_data_path>)), <prediction_data_path>)
```

A full example with saving, loading & some dummy data can be found [here](https://github.com/98mprice/pyAudioClassification/blob/master/example/test.py)

Read below for a more detailed look at each of these calls.

## Detailed Guide
### Step 1: Preprocessing 🐶 🐱
First, add all your audio files to a directory in the following structure
```
data/
├── <class_name>/
│   ├── <file_name>
│   └── ...
└── ...
```

For example, if you were trying to classify dog and cat sounds it might look like this
```
data/
├── cat/
│   ├── cat1.ogg
│   ├── cat2.ogg
│   ├── cat3.wav
│   └── cat4.wav
└── dog/
    ├── dog1.ogg
    ├── dog2.ogg
    ├── dog3.wav
    └── dog4.wav
```

Great, now we need to preprocess this data. Just call `feature_extraction(<data_path>)` and it'll return our input and target data.
Something like this:
```python
features, labels = feature_extraction('/Users/mac2015/data/')
```

(If you don't want to print to stdout, just pass `verbose=False` as a argument)

---
Depending on how much data you have, this process could take a while... so it might be a good idea to save. You can save and load with [NumPy](https://www.numpy.org/)
```python
np.save('%s.npy' % <file_name>, features)
features = np.load('%s.npy' % <file_name>)
```

### Step 2: Training 💪
Next step is to train your model on the data. You can just call...
```python
model = train(features, labels)
```
...but depending on your dataset, you might need to play around with some of the hyper-parameters to get the best results.

#### Options
* `epochs`: The number of iterations. Default is `50`.

* `lr`: Learning rate. Increase to speed up training time, decrease to get more accurate results (if your loss is 'jumping'). Default is `0.01`.

* `optimiser`: Choose any of [these](https://keras.io/optimizers/). Default is `'SGD'`.

* `print_summary`: Prints a summary of the model you'll be training. Default is `False`.

* `loss_type`: Classification type. Default is `categorical` for >2 classes, and `binary` otherwise.

You can add any of these as optional arguments, for example `train(features, labels, lr=0.05)`

---
Again, you probably want to save your model once it's done training. You can do this with Keras:
```python
from keras.models import load_model

model.save('my_model.h5')
model = load_model('my_model.h5')
```

### Step 3: Prediction 🙏 🙌
Now the fun part- try your trained model on new data!

```python
pred = predict(model, <data_path>)
```

Your `<data_path>` should point to a new, untested audio file.

#### Binary
If you have 2 classes (or if you force selected `'binary'` as a type), `pred` will just be a single number for each file.

The closer it is to 0, the closer the prediction is for the first class, and the closer it is to 1 the closer the prediction is to the second class.

So for our cat/dog example, if it returns `0.2` it's 80% sure the sound is a cat, and if it returns `0.8` it's 80% sure it's a dog.

#### Categorical
If you have more than 2 classes (or if you force selected `'categorical'` as a type), `pred` will be an array for each sound file.

It'll look something like this
```
[[1.6454633e-06 3.7017996e-11 9.9999821e-01 1.5900606e-07]]
```

The index of each item in the array will correspond to the prediction for that class.

---
You can pretty print the predictions by showing them in a leaderboard, like so:

```python
print_leaderboard(pred, <training_data_path>)
```
It looks like this:

```
1. Cow 100.0% (index 2)
2. Rooster 0.0% (index 0)
3. Frog 0.0% (index 3)
4. Pig 0.0% (index 1)
```

## References
* Large parts of the code (particularly the feature extraction) are based on [mtobeiyf/audio-classification](https://github.com/mtobeiyf/audio-classification)
* [panotti](https://github.com/drscotthawley/panotti)
