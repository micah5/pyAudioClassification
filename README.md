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

## Quick start
```python
from pyaudioclassification import feature_extraction, train, predict
features, labels = feature_extraction(<data_path>)
model = train(features, labels)
pred = predict(model)
```

Or, if you're feeling reckless, you could just string them together like so:
```python
pred = predict(train(feature_extraction(<data_path>)))
```

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
---
Depending on how much data you have, this process might take a while... so it could be a good idea to save the processed data. For this purpose I added some small wrapper functions:
```python
save_features(features, labels)
features, labels = load_features(<features_path>, <labels_path>)
```
You can also specify the names of these files by passing them in a `file_names` argument tuple, but they'll default to 'feat' and 'label'.

Also, you can automatically save in your `feature_extraction` call by passing `save=True`

### Step 2: Training 💪
Next step is to train your model on the data. You can just call...
```python
model = train(features, labels)
```
...but depending on your dataset, you might need to play around with some of the options to get the best results.
