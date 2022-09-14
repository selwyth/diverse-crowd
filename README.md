### Installation

`pipenv install` if using pipenv

OR

`pip install -r requirements.txt` if not

### Run

`python train.py --help` for options

Use `python train.py --filename test --refresh` as an example.

--filename specifies what name to save the files as. These files will all be saved to the `static` folder with the following convention:
- <filename> : a binary file of tweets in raw-text form
- <filename>.model : a gensim-model pickle, this is saved but not used later (but can be used later if you want to feed more raw data into the model)
- <filename>.kv : a gensim-model KeyedVector instance that is loaded in order to calculate vectors from input words

--refresh / --no-refresh is an option that speeds things up and saves money by calling from the local file instead of the Twitter API. If you put --refresh, it will definitely call Twitter's API; if you put --no-refresh, it'll scan your `static` folder for <filename>, and if available, load it up to proceed; if not available, it will do the same as --refresh.

--word_vectors is an option that can be supplied in addition to --filename. Setting this option with a model name provided [here](https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models) will result in the corresponding model being downloaded from gensim, instead of training a model using tweets from Twitter API or cached data. The tweets data will still be used to calculate user similarity etc, but not used to train a model.

Example: `python train.py --word_vectors glove-twitter-25` --filename test

### Model Development Ideas

1. More users, more tweets beyond most recent 20
2. Use bigrams
3. Tweaking gensim hyperparameters e.g. min_count
4. Clean out more stopwords like 'and', 'or', 'I'll'
5. Use representative words, phrases and tweets instead of representative users
6. Label dimensions like liberal/conservative, pro-life/pro-choice, crypto/anti-crypto
