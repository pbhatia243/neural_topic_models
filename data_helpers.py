import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.python.platform import gfile
import codecs
import sys
from spacy.en import English
import re
from nltk.stem import WordNetLemmatizer
import os

import sys
reload(sys)
sys.path.insert(0, '../')
# from topic_models.utils import spacy_tokenize_data
_WORD_SPLIT = re.compile("([.,!/?\":;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary_data(vocabulary_path, data, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, vocabulary_path))
    vocab = {}

    counter = 0
    for line in data:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        # data =line.rstrip()
        # txt  = data
        # tokens = tokenizer(txt) if tokenizer else basic_tokenizer(txt)
        for w in line:
          # word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          word = w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
    vocab_list =  sorted(vocab, key=vocab.get, reverse=True)
    print len(vocab_list)
    if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        data =line.rstrip()
        txt  = data
        tokens = tokenizer(txt) if tokenizer else basic_tokenizer(txt)
        for w in tokens:
          # word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          word = w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list =  sorted(vocab, key=vocab.get, reverse=True)
      print len(vocab_list)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

def read_chat_data(data_path,vocabulary_path, max_size=10000000):
    counter = 0
    vocab, _ = initialize_vocabulary(vocabulary_path)
    data_set = []
    with codecs.open(data_path, "rb") as fi:
        for line in fi.readlines():
            counter += 1
            if max_size!=0 and counter > max_size:
                break
            if counter % 10000 == 0:
              print("  reading data line %d" % counter)
              sys.stdout.flush()
            # entities = line.split("\t")
            # print entities
            txt  = line
            source_ids = [int(x) for x in sentence_to_token_ids(txt,vocab)]
            data_set.append(source_ids)

    return data_set

def read_data(data,vocabulary_path, max_size=10000000):
    counter = 0
    vocab, _ = initialize_vocabulary(vocabulary_path)
    data_set = []

    for line in data:
            counter += 1
            if max_size!=0 and counter > max_size:
                break
            if counter % 10000 == 0:
              print("  reading data line %d" % counter)
              sys.stdout.flush()
            # entities = line.split("\t")
            # print entities
            txt  = line
            sen2token = []
            for w in txt:
                if w in vocab:
                    sen2token.append(int(vocab.get(w)))
            data_set.append(sen2token)

    return data_set

def generate( fn, pos_mode = 2):
         nlp = English()

         lmtzr = WordNetLemmatizer()
         ncorpus =[]
         if pos_mode == 0:
             print "Nouns tags"
             tags = ["NN" ,  "NNS", "NNP", "NNPS"]
         elif pos_mode == 1:
             print "Verbs"
             tags = ["VB"]
         else:
             print "Mix"
             tags = ["NN" ,  "NNS", "NNP", "NNPS", "VB"]

         i=0
         all = []
         print "Started"
         stopwords = ["fuck","haha","lol", "op", "sex", "someone", "nothing", "dick", "bitch", "pussy", "porn", "cunt", "bitch", "kik"]
         with codecs.open( os.path.join(fn), 'rb')  as fi:
                for line in fi.readlines():
                    for stopword in stopwords:
                        line = line.replace(stopword, " ")

                    words, data_u = Utils.spacy_tokenize_data(line, nlp, lmtzr, tags)

                    i+=1
                    if i%1000==0:
                        print str(i) ,  " datas processed so far"
                    all.append(words)

         return all


def generate_simple(fn):
    i=0
    all =[]
    with codecs.open(os.path.join(fn), 'rb')  as fi:
        for line in fi.readlines():


            words= line.rstrip().split(' ')

            i += 1
            if i % 1000 == 0:
                print str(i), " sentences processed so far"
            all.append(words)

    return all

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  # if not normalize_digits:
  sen2token = []
  for w in words:
      if w in vocabulary:
          sen2token.append(vocabulary.get(w))

  return sen2token
  # Normalize digits by 0 before looking words up in the vocabulary.
  # return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/vocab_generator.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)+1
    print num_batches_per_epoch
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index-start_index < batch_size:
                # print batch_size
                remaining  = batch_size - (end_index-start_index)
                # print remaining
                new_end_index = remaining

                yield np.concatenate((shuffled_data[start_index:end_index],shuffled_data[0:new_end_index]), axis=0)
            else:
                yield shuffled_data[start_index:end_index]


def batch_iter_brnn(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)+1
    print num_batches_per_epoch
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index-start_index < batch_size:
                print batch_size
                remaining  = batch_size - (end_index-start_index)
                print remaining
                new_end_index = remaining

                yield np.concatenate((shuffled_data[start_index:end_index],shuffled_data[0:new_end_index]), axis=0)
            else:
                yield shuffled_data[start_index:end_index]
def batch_iter_simple(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)+1
    print num_batches_per_epoch
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index-start_index < batch_size:
                print batch_size
                remaining  = batch_size - (end_index-start_index)
                print remaining
                new_end_index = remaining

                yield np.concatenate((shuffled_data[start_index:end_index],shuffled_data[0:new_end_index]), axis=0)
            else:
                yield shuffled_data[start_index:end_index]