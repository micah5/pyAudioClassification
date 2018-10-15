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
```
from pyaudioclassification import feature_extraction, train, predict
features, labels = feature_extraction(<data_path>)
model = train(features, labels)
pred = predict(model)
```

Or, if you're feeling reckless, you could just string them together like so:
```
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
```
features, labels = feature_extraction('/Users/mac2015/data/')
```
