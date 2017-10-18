@title[Introduction]

# Introduction to Keras

### Machine Learning PyVo 2017

### Petr Baudis, Rossum

https://github.com/rossumai/pyvo17-imdb

---
@title[Basic Task]

### IMDb Reviews Sentiment

this is one amazing movie!!!!! you have to realize that chinese folklore is complicated and philosophical. there are always stories behind stories. i myself did not understand everything but knowing chinese folklore (i studied them in school)it is very complicated. you just have to take what it gives you.....ENJOY THE MOVIE AND ENJOY THE RIDE....HOORAY!!!!

I think I will make a movie next weekend. Oh wait, I'm working..oh I'm sure I can fit it in. It looks like whoever made this film fit it in. I hope the makers of this crap have day jobs because this film sucked!!! It looks like someones home movie and I don't think more than $100 was spent making it!!! Total crap!!! Who let's this stuff be released?!?!?!

---
@title[Training Problem]

#### Goal: Predict sentiment from text input

We have labelled examples (_dataset_ - _supervised_ training).

Labels are yes/no - _categorical_.

*Machine Learning Model:* Propose a mathematical formula that computes the sentiment from input.

*Machine Learning Training:* Find coefficients in the mathematical formula automatically.

How to encode input mathematically?

---
@title[Bag-of-Words Representation]

#### Can we guess the sentiment based on isolated words?

this is one *amazing* movie!!!!! you have to realize that chinese folklore is _complicated_ and philosophical.

_hope_ the makers of this *crap* have day jobs because this film *sucked*!!!

---
@title[One-hot Encoding]

#### How to represent words mathematically?

*One-hot encoding:* An array as big as dictionary, all zeroes except a single 1 at the index of the word.

To determine sentiment: Multiply each element by the _word weight_ (positive or negative), sum them up.

amazing, complicated, hope, crap, sucked

---
@title[Text to Words]

First problem in Natural Language Processing: *Tokenization*

this is one amazing movie!!!!! you have to realize that chinese folklore is complicated and philosophical. there are always stories behind stories. i myself did not understand everything but knowing chinese folklore (i studied them in school)it is very complicated. you just have to take what it gives you.....ENJOY THE MOVIE AND ENJOY THE RIDE....HOORAY!!!!

---
@title[Text to Words]

First problem in Natural Language Processing: *Tokenization*

this is one amazing movie!!!!! you have to realize that chinese folklore is complicated and philosophical. there are always stories behind stories. i myself did not understand everything but knowing chinese folklore (i studied them in school)it is very complicated. you just have to take what it gives you.....ENJOY THE MOVIE AND ENJOY THE RIDE....HOORAY!!!!

```python
def text_tokens(text):
    text = text.lower()
    text = re.sub("\\s", " ", text)
    text = re.sub("[^a-zA-Z' ]", "", text)
    tokens = text.split(' ')
    return tokens
```

---
@title[Text Encoding]

```python
# Vocabulary: All words used in reviews
with open('aclImdb/imdb.vocab') as f:
    vocab = [word.rstrip() for word in f]

def review_bow_vector(tokens):
    vector = [0] * len(vocab)
    for t in tokens:
        try:
            vector[vocab.index(t)] = 1
        except:
            pass  # ignore missing words
    return vector
```

---
@title[Keras Framework]

*Neural Networks* are just mathematical formulas like the above.

Only typically more complicated.

The formula is a *Model*, with some *input* and *output* variables.

```python
class SentimentModel(object):
    def __init__(self):
        bow = Input(shape=(len(vocab),), name='bow_input')
        # the actual formula: give weights to all elements and sum once
        sentiment = Dense(1)(bow)
        # normalize to [0, 1] range
        sentiment = Activation('sigmoid')(sentiment)

        self.model = Model(input=bow, output=sentiment)
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train(self, X, y, X_val, y_val):
        self.model.fit(X, y, validation_data=(X_val, y_val),
                       epochs=25, verbose=1)
```

---
@title[Training Keras Model]

```python
def load_dataset(dirname):
    X, y = [], []
    # Review files: neg/0_3.txt neg/10000_4.txt neg/10001_4.txt ...
    for y_val, y_label in enumerate(['neg', 'pos']):
        y_dir = os.path.join(dirname, y_label)
        for fname in os.listdir(y_dir):
            fpath = os.path.join(y_dir, fname)
            with open(fpath) as f:
                tokens = text_tokens(f.read())
            bow = review_bow_vector(tokens)
            X.append(bow)
            y.append(y_val)  # 0 for 'neg', 1 for 'pos'
    return np.array(X), np.array(y)

X_train, y_train = load_dataset('aclImdb/train/')
X_val, y_val = load_dataset('aclImdb/test/')
sentiment = SentimentModel()
sentiment.train(X_train, y_train, X_val, y_val)
```

---
@title[Running This]

```
Epoch 1/25
25000/25000 [==============================] - 3s - loss: 0.4550 - acc: 0.8307 - val_loss: 0.3629 - val_acc: 0.8736
...
Epoch 4/25
25000/25000 [==============================] - 3s - loss: 0.2503 - acc: 0.9095 - val_loss: 0.2956 - val_acc: 0.8811
Epoch 5/25
25000/25000 [==============================] - 3s - loss: 0.2350 - acc: 0.9130 - val_loss: 0.2914 - val_acc: 0.8810
...
Epoch 20/25
25000/25000 [==============================] - 3s - loss: 0.1652 - acc: 0.9390 - val_loss: 0.3373 - val_acc: 0.8674
...
Epoch 25/25
25000/25000 [==============================] - 3s - loss: 0.1569 - acc: 0.9420 - val_loss: 0.3565 - val_acc: 0.8636
Good story about a backwoods community in the Ozarks around the turn of the century. Moonshine is the leading industry, fighting and funning the major form of entertainment. One day a stranger enters the community and causes a shake-up among the locals. Beautiful scenery adds much to the story. [ 0.86337662]
```

---
@title[Take-aways]

Neural networks are just mathematical formulas.
Training is finding the right coefficients.

Pick the simplest model, stick with it if it's good enough.

80% of the work is the data, not the models.

#### Thanks for your attention!