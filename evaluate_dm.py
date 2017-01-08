#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from data_helpers import *
from models import VariationalTopicModelBatch
import cPickle
from vocab_generator import DataBuilder
# Parameters
# ==================================================
from sklearn.metrics import matthews_corrcoef
# Model Hyperparameters
import argparse
import codecs
from scipy import spatial
import sys
reload( sys)

sys.setdefaultencoding( 'utf-8')
import sys
reload(sys)
sys.path.insert(0, '../')
sys.setdefaultencoding('utf-8')
from topic_models.utils import *

def load_model(checkpoint_dir, model_dir, latent_dim):
        with tf.Graph().as_default():
              session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
              sess = tf.Session(config=session_conf)
              with sess.as_default():


                    nvdm = VariationalTopicModelBatch(
                        sess=sess,
                        vocab_size=en_vocab_size,
                        hidden_encoder_dim=500,
                        latent_dim=latent_dim)

                    saver = tf.train.Saver(tf.all_variables())
                    tf.initialize_all_variables().run()

                    print(" [*] Loading checkpoints...")

                    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
                        print(" [*] Load SUCCESS")
                        return sess ,  nvdm
                    else:
                        print(" [!] Load failed...")
                        return sess , nvdm

def save(self, checkpoint_dir, global_step=None):
        self.saver = tf.train.Saver()

        print(" [*] Saving checkpoints...")
        model_name = "variational_dm_100k"
        model_dir = self.get_model_dir()

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name), global_step=global_step)
        return checkpoint_dir

def get_model_dir(self):
        return self.model_dir


def test_topic_words(sess, nvdm, rev_voc):
    r_val = sess.run(nvdm.R)
    top_indices = list()
    num_top_indices_per_dim = 10
    num_dims =25
    for i in xrange(num_dims):
        top_indices.append(np.argsort(np.array([-val[i] for val in r_val])))
    for i in xrange(num_top_indices_per_dim):
        for j in xrange(num_dims):
            print rev_voc[top_indices[j][i]],
        print ''


def similarity( sess, nvdm, vocab, rev_voc, num=10):
    eval_text = ["trump", "pizza", "republic", "food", "love", "guy"]
    r_val = sess.run(nvdm.R)
    eval_features = []
    for word in eval_text:
        eval_features.append(r_val[vocab[word]])


    lat_rep = np.array(r_val)

    for query, text in zip(eval_features, eval_text):
        print text, "=====>>>>"
        dist = []
        # query_vector= query
        query_vector = query[np.newaxis, :]
        dist.append(spatial.distance.cdist(query_vector, lat_rep))
        dist = np.hstack(dist)
        ranked = np.squeeze(dist.argsort())[:num]
        for ranks in ranked:
            print rev_voc[ranks]

def similarity( sess, nvdm, vocab, rev_voc, num=10):
    eval_text = ["trump", "pizza", "republic", "food", "love", "guy"]
    r_val = sess.run(nvdm.R)
    eval_features = []
    for word in eval_text:
        eval_features.append(r_val[vocab[word]])


    lat_rep = np.array(r_val)

    for query, text in zip(eval_features, eval_text):
        print text, "=====>>>>"
        dist = []
        # query_vector= query
        query_vector = query[np.newaxis, :]
        dist.append(spatial.distance.cdist(query_vector, lat_rep))
        dist = np.hstack(dist)
        ranked = np.squeeze(dist.argsort())[:num]
        for ranks in ranked:
            print rev_voc[ranks]

def get_eval(vocab):
    yaks = []
    indxs = []
    yak_text = []
    with codecs.open("some_yaks.txt", 'rb')  as fi:
        for line in fi.readlines():
            yaks.append(line.rstrip().lower())
    eval_text = generate_data(yaks)
    final_yaks = []
    for i, line in enumerate(eval_text):
        sen2token = []
        for w in line:
            if w in vocab:
                sen2token.append(vocab.get(w))
        if len(sen2token) > 1 :
            final_yaks.append(line)
            indxs.append(sen2token)
            yak_text.append(yaks[i])
    return final_yaks, indxs, yak_text

def doc_similarity(sess, nvdm, vocab):
    eval_text = ["need to go to gym and workout", "i can not tolerate racism and stereotypes",
                 "i dont want to go for republicans or democrats :P",
                 "football season is on ", "any good places to eat pizza ?",
                 "breaking bad the best tv series ever", "women empowerment or feminism ?", "I want to eat pizza"]
    eval_text = generate_data(eval_text)
    lat_rep_list = []
    final_yaks, indxs, yak_text = get_eval(vocab)
    for word_idxs in indxs:
        # print word_idxs
        x = np.bincount(list(word_idxs), minlength=len(vocab))
        x = x[np.newaxis, :]
        latent_representation = sess.run(nvdm.h, feed_dict={nvdm.x: x})
        lat_rep_list.append(latent_representation[0])

    eval_features = []
    for txt in eval_text:
        sen2token = []
        for w in txt:
            if w in vocab:
                sen2token.append(vocab.get(w))

        x = np.bincount(list(sen2token), minlength=len(vocab))
        x = x[np.newaxis, :]
        latent_representation = sess.run(nvdm.h, feed_dict={nvdm.x: x})
        eval_features.append(latent_representation[0])

    lat_rep = np.array(lat_rep_list)

    for query, text in zip(eval_features, eval_text):
        print text, "=====>>>>"
        dist = []
        # query_vector= query
        query_vector = query[np.newaxis, :]
        dist.append(spatial.distance.cdist(query_vector, lat_rep))
        dist = np.hstack(dist)
        ranked = np.squeeze(dist.argsort())[:10]
        for ranks in ranked:
            print final_yaks[ranks]
            print yak_text[ranks]

def topic_similarity(sess, nvdm, vocab, rev_voc):


    lat_rep_list = []
    for word in vocab:
        sen2token = []

        sen2token.append(vocab.get(word))

        x = np.bincount(list(sen2token), minlength=len(vocab))
        x = x[np.newaxis, :]
        latent_representation = sess.run(nvdm.h, feed_dict={nvdm.x: x})
        lat_rep_list.append(latent_representation[0])

    lat_rep = np.array(lat_rep_list)

    top_indices = list()
    num_top_indices_per_dim = 25
    num_dims = 25
    for i in xrange(num_dims):
        top_indices.append(np.argsort(np.array([-val[i] for val in lat_rep])))
    for j in xrange(num_top_indices_per_dim):
        print "Topic , ", j
        for i in xrange(num_dims):
            print rev_voc[top_indices[i][j]],
        print ''


def savemodel(fname,D):
            """ Save model into fname
            """
            if not fname.endswith('.pickle.gz'):
                fname = fname + '.pickle.gz'
            # D = self.getparams()
            with gzip.open(fname, 'w') as fout:
                dump(D, fout)
            print 'Save model into file {}'.format(fname)
if __name__=="__main__":

    #0 for train 1 for evaluation 2 for prediction
    mode = 0
    en_vocab_size = 40000
    parser = argparse.ArgumentParser( description = 'CNN Sentiment')
    vocab_path = "data/vocab_80k.en"
    # data_path = "All_100000_100k_data.txt"
    checkpoint_dir = "model_checkpoints/"
    model_dir = "yikyak_data/"
    latent_dim = 200
    vocab, rev_voc = initialize_vocabulary(vocab_path)
    sess, nvdm = load_model(checkpoint_dir, model_dir, latent_dim)
    r_val = sess.run(nvdm.R)
    D = {}
    D["embed"] =r_val
    D["vocab"] = vocab
    D["rev_voc"] = rev_voc
    Utils.savemodel("document_model_80k", D)

    #
    # def savemodel(fname, D):
    #     """ Save model into fname
    #     """
    #     if not fname.endswith('.pickle.gz'):
    #         fname = fname + '.pickle.gz'
    #     # D = self.getparams()
    #     with gzip.open(fname, 'w') as fout:
    #         dump(D, fout)
    #     print 'Save model into file {}'.format(fname)
    # get_embedding(sess, nvdm)
    # yaks, indxs = get_eval(vocab)
    # doc_similarity(sess, nvdm, vocab)
    # test_topic_words(sess, nvdm, rev_voc)
    similarity(sess, nvdm, vocab, rev_voc)
    # word_similarity(sess, nvdm, vocab, rev_voc, words)
    # topic_similarity(sess, nvdm, vocab, rev_voc)
