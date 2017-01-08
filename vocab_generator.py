import numpy as np
from cPickle import load, dump
from collections import defaultdict, Counter
import sys, re
import pandas as pd
import gzip
import cPickle
import codecs

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_UNK_LOC = ["_UNK_LOC"]
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
class DataBuilder():

    def __init__(self):
       self.dc =  DataCleaner()

    def build_train_data(self,data_folder, cv=10, clean_string=False):
        """
        Loads data and split into 10 folds by default.
        """
        revs = []

        vocab = defaultdict(float)
        print data_folder
        with codecs.open( data_folder, 'rb')  as fi:
            for line in fi.readlines():
                line = line.decode('utf-8')
                parts = line.split("\n")[0].split("\t")
                if len(parts) > 1:
                    sent = parts[1]
                    rev = []
                    rev.append(sent.strip())

                    if clean_string:
                        orig_rev = self.dc.clean_str(" ".join(rev))
                    else:
                        orig_rev = " ".join(rev).lower()
                    #print orig_rev
                    words = set(orig_rev.split())
                    for word in words:
                            vocab[word.lower()] += 1
                    if len(orig_rev.split()) < 50 :

                            datum  = {"y":int(parts[0]),
                                      "text": orig_rev,
                                      "num_words": len(orig_rev.split()),
                                      "split": np.random.randint(0,cv)}
                            revs.append(datum)
                # else:
                #     print orig_rev


        return revs, vocab

    def build_eval_data(self,data, num_classes, clean_string=False):
        """
        Loads data and split into 10 folds by default.
        """
        revs = []
        num = [-1, 1]
        for line in data:
                line = line.decode('utf-8')
                sent = line
                rev = []
                rev.append(sent.strip())

                if clean_string:
                    orig_rev = self.dc.clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                #print orig_rev

                datum  = {"y":num[np.random.randint(0, num_classes)],
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          }
                revs.append(datum)


        return revs

    def build_data_cv(self, data_folder, cv=10, clean_string=True):
        """
        Loads data and split into 10 folds.
        """
        revs = []
        pos_file = data_folder[0]
        neg_file = data_folder[1]
        vocab = defaultdict(float)
        with open(pos_file, "rb") as f:
            for line in f:
                rev = []
                rev.append(line.strip())
                if clean_string:
                    orig_rev = self.dc.clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":1,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)
        with open(neg_file, "rb") as f:
            for line in f:
                rev = []
                rev.append(line.strip())
                if clean_string:
                    orig_rev = self.dc.clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":0,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)
        return revs, vocab

    def build_data(self, data_folder, cv=10, clean_string=False):
        """
        Loads data and split into 10 folds.
        """
        revs = []
        # pos_file = loadmodel(data_folder[0])
        # neg_file = loadmodel(data_folder[1])
        pos_yaks = loadmodel(data_folder[0]).get("content")
        neg_yaks = loadmodel(data_folder[1]).get("content")
        vocab = defaultdict(float)
        happyList = [ ":-)", ":)", ":D", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", ":?)",   ":-)", ": )", ": D", ": o)", ":]", ": 3", ":c)", ":>", "= ]", "8 )", "= )", ": }", ":^)", ":?)"   ]
        sadList = [ ">:[", ":-(", ":(", ":-c", ":c", ":-<", ":?C", ":<", ":-[", ":[", ":{",">:[", ":-(", ": (", ":-c", ": c", ": -<", ": ?C", ": <", ": -[", ": [", ": {" ]
        for line in pos_yaks:
                rev = []
                rev.append(line.strip())

                if clean_string:
                    orig_rev = self.dc.clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                #print orig_rev
                words = set(orig_rev.split())
                for word in words:
                    if word in happyList or word in sadList:
                        pass
                    else:
                        vocab[word] += 1
                datum  = {"y":1,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)

        for line in neg_yaks:
                rev = []
                rev.append(line.strip())
                if clean_string:
                    orig_rev = self.dc.clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    if word in happyList or word in sadList:
                        pass
                    else:
                        vocab[word] += 1
                datum  = {"y":0,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)
        return revs, vocab


    def get_idx_from_sent(self, sent, word_idx_map, max_l=45, k=300, filter_h=5):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        pad = filter_h - 1
        # for i in xrange(pad):
        #     x.append(0)
        words = sent.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
                if len(x)==max_l+pad:
                    break
        while len(x) < max_l+2*pad:
             x.append(0)
        return x

    def get_idx_from_sent_rnn(self, sent, word_idx_map, max_l=45):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        mask = None
        pad =1
        for i in xrange(pad):
            x.append(0)

        words = sent.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
                if len(x)==max_l+pad:
                    break
        mask =  len(x)
        while len(x) < max_l+2*pad:
            x.append(0)

        return x, mask

    def get_idx_from_sent_brnn(self, sent, word_idx_map, max_l=45):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        rev_x = []
        mask = None
        pad =1
        for i in xrange(pad):
            x.append(0)
            rev_x.append(0)

        words = sent.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
                rev_x.append(word_idx_map[word])
                if len(x)==max_l+pad:
                    break
        mask =  len(x)
        rev_x.append(0)
        rev_x.reverse()
        while len(rev_x) < max_l+2*pad:
            rev_x.append(0)
        while len(x) < max_l+2*pad:
            x.append(0)

        return x,rev_x, mask

    def get_idx_from_sent_rnn_mask(self, sent, word_idx_map, max_l=45, k=300, filter_h=5):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        mask = None
        pad = 1
        for i in xrange(pad):
            x.append(0)

        words = sent.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
                mask.append(1)
        while len(x) < max_l+2*pad:
            x.append(0)
            mask.append(0)
        return x, mask
    def make_idx_data_cv(self, revs, word_idx_map, cv, max_l=59, k=300, filter_h=5):
        """
        Transforms sentences into a 2-d matrix.
        """
        train, test = [], []
        for rev in revs:
            sent = self.get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
            sent.append(rev["y"])
            if rev["split"]==cv:
                test.append(sent)
            else:
                train.append(sent)
        train = np.array(train,dtype="int")
        test = np.array(test,dtype="int")
        return [train, test]


class Distributional_Representation():

    def __init__(self):
        None

    def get_W(self,word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
        W[0] = np.zeros(k, dtype='float32')
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
            """
            Loads 300x1 word vecs from Google (Mikolov) word2vec
            """
            word_vecs = {}
            with open(fname, "rb") as f:
                header = f.readline()
                print header
                vocab_size, layer1_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer1_size
                for line in xrange(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)
                    if word in vocab:
                       word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                       # logger.info(word_vecs[word])
                    else:
                        f.read(binary_len)
           # logger.info("num words already in word2vec: " + str(len(word_vecs)))
            return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=3, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)

class  DataCleaner():

    def __init__(self):
        None

    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
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

    def clean_str_sst(string):
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

def loadmodel(fname):
            """ Load model from fname
            """
            if not fname.endswith('.pickle.gz'):
                fname = fname + '.pickle.gz'
            with gzip.open(fname, 'r') as fin:
                D = load(fin)
            print 'Load model from file: {}'.format(fname)
            return D
class Build_Model_Data():
    def __init__(self):
        None

    def test_run(self, w2v_file, data_file, save_file,max_vocabulary_size = 80000):

        print "loading data...",
        db = DataBuilder()
        revs, vocab = db.build_train_data(data_file, cv=10, clean_string=False)
        # vocab =  vocab.most_common(max_vocabulary_size)
        # if len(vocab_list) > max_vocabulary_size:
        #     vocab = vocab_list[:max_vocabulary_size]
        max_l = np.max(pd.DataFrame(revs)["num_words"])
        print "data loaded!"
        print "number of sentences: " + str(len(revs))
        print "vocab size: " + str(len(vocab))
        print "max sentence length: " + str(max_l)
        print "loading word2vec vectors...",

        dist_rep =  Distributional_Representation()
        w2v = dist_rep.load_bin_vec(w2v_file, vocab)
        print "word2vec loaded!"
        print "num words already in word2vec: " + str(len(w2v))
        dist_rep.add_unknown_words(w2v, vocab)
        W, word_idx_map = dist_rep.get_W(w2v)
        print len(W)
        print word_idx_map
        # print "Complete Random Representation"
        # rand_vecs = {}
        # dist_rep.add_unknown_words(rand_vecs, vocab)
        # R, _ = dist_rep.get_W(rand_vecs)
        cPickle.dump([revs, W, W , word_idx_map, vocab, max_l], open(save_file, "wb"))
        print "dataset created!"

if __name__=="__main__":
    max_vocabulary_size = 80000
    w2v_file = "/Users/Parry/Downloads/GoogleNews-vectors-negative300.bin"
    data_file = "data/all_final_kk.data.txt"
    save_file = "test.p"
    bm =  Build_Model_Data()
    bm.test_run(w2v_file, data_file, save_file,max_vocabulary_size = max_vocabulary_size)

    
