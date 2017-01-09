x#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from data_helpers import *
from models import VariationalTopicModelBatch
import cPickle
# from vocab_generator import DataBuilder
# Parameters
# ==================================================

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
# from topic_models.utils import *
class Neural_DM():

    def __init__(self,batch_size, model_dir, latent_dim = 100):

        self.l2_reg_lambda= 0.0
        self.batch_size = batch_size
        self.num_epochs=  30
        self.evaluate_every =200
        self.checkpoint_every = 400
        self.allow_soft_placement = True
        self.log_device_placement = False
        self.checkpoint_dir = "model_checkpoints/"
        self.model_dir = model_dir
        self.decay_rate=0.9
        self.decay_step=5000
        self.learning_rate=0.001
        self.latent_dim = latent_dim




    # Data Preparation
    # ==================================================

    # Training
    # ==================================================
    def train_model_batch(self, train_set, en_vocab_size, vocab, rev_vocab ):

        print("Vocabulary Size: {:d}".format(en_vocab_size))
        print("Train/Dev split: {:d}/{:d}".format(len(train_set), len(train_set)))
        # texts, indxs = self.get_eval(vocab)
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=self.allow_soft_placement,
              log_device_placement=self.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.step = tf.Variable(0, trainable=False)
                global_step = tf.Variable(0, name="global_step", trainable=False)
                self.lr = tf.train.exponential_decay(self.learning_rate, global_step, 10000, self.decay_rate, staircase=True, name="lr")
                _ = tf.scalar_summary("learning rate", self.lr)

                self.nvdm = VariationalTopicModelBatch(
                    sess=self.sess,
                    vocab_size=en_vocab_size,
                    hidden_encoder_dim=500,
                    latent_dim=self.latent_dim, model_dir=self.model_dir)


                # Define Training procedure

                # train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.nvdm.loss, global_step=global_step)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                grads_and_vars = optimizer.compute_gradients(self.nvdm.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                self.saver = tf.train.Saver(tf.all_variables())
                tf.initialize_all_variables().run()
                self.load(self.checkpoint_dir)
                # Initialize all variables
                def train_step(x_batch):
                    """
                    A single training step
                    """
                    # print len(x_batch)
                    x_bins = []
                    for x_in in x_batch:
                        x = np.bincount(list(x_in), minlength=en_vocab_size)
                        x_bins.append(x)
                    x_bins = np.array(x_bins)

                    # print x_bins.shape
                    feed_dict = {
                      self.nvdm.x: x_bins

                    }
                    _, step, loss , kl , ll, summary_str= self.sess.run(
                        [train_op, global_step, self.nvdm.loss, self.nvdm.kl_divergence, self.nvdm.likelihood, self.nvdm.merged_sum],
                        feed_dict)
                    return loss , kl , ll, summary_str, step

                # print train_set
                batches = data_helpers.batch_iter(
                train_set, self.batch_size, self.num_epochs)
                     # Training loop. For each batch...
                for batch in batches:
                        # print "New Batch started !!!!!!!!!!!!!!"
                        x_batch = batch
                        loss , kl , ll, summary_str, step = train_step(x_batch)
                        current_step = tf.train.global_step(self.sess, global_step)

                        if current_step % self.evaluate_every == 0:
                            print("Evaluation:")
                            print("Step: [%4d] , loss: %.8f" \
                                 % (current_step,  loss))

                        if current_step % 20 == 0:
                            self.nvdm.writer.add_summary(summary_str, current_step)

                        if current_step % 5000 == 0:
                                save_path = self.save(self.checkpoint_dir,step)
                                print "Model saved in file: ", save_path


    def train(self, train_set, en_vocab_size, vocab, rev_vocab):
        # To add other neural topic models here
            self.train_model_batch( train_set, en_vocab_size, vocab, rev_vocab )

    def get_eval(self, vocab):
        texts = []
        indxs = []
        with codecs.open( "some_texts.txt", 'rb')  as fi:
                for line in fi.readlines():
                    texts.append(line.rstrip().lower())
                    indxs.append(sentence_to_token_ids(line, vocab))
        return texts, indxs

    def similarity(self, vocab, indxs, texts):
         eval_text = ["Need to go to gym and workout", "I can not tolerate racism and stereotypes",
                 "I dont want to go for republicans or democrats :P",
                 "Football season is on ", "Any good places to eat pizza ?",
                 "Breaking Bad the best TV series ever", "Women empowerment or Feminism ?", "I want to eat pizza" ]
         lat_rep_list = []
         for word_idxs in indxs:
            x = np.bincount(list(word_idxs), minlength=len(vocab))
            latent_representation = self.sess.run(self.nvdm.h, feed_dict={self.nvdm.x: x})
            lat_rep_list.append(latent_representation[0])

         eval_features = []
         for txt in eval_text:
            word_idxs =  sentence_to_token_ids(txt.lower(), vocab)
            x = np.bincount(list(word_idxs), minlength=len(vocab))
            latent_representation = self.sess.run(self.nvdm.h, feed_dict={self.nvdm.x: x})
            eval_features.append(latent_representation[0])

         lat_rep = np.array(lat_rep_list)

         for query, text in zip(eval_features,eval_text):
            print text , "=====>>>>"
            dist = []
            # query_vector= query
            query_vector= query[np.newaxis,:]
            dist.append(spatial.distance.cdist(query_vector,lat_rep))
            dist = np.hstack(dist)
            ranked = np.squeeze(dist.argsort())[:10]
            for ranks in ranked:
                print texts[ranks]


    def sample(self, vocab, rev_vocab, sample_size=20, text=None):
        """Sample the documents."""
        p = 1

        if text != None:
          try:
            word_idxs = sentence_to_token_ids(text, vocab)
            x = np.bincount(list(word_idxs), minlength=len(vocab))
          except Exception as e:
            print(e)
            return

        print(" [*] Text: %s" % " ".join([rev_vocab[word_idx] for word_idx in word_idxs]))

        cur_ps = self.sess.run(self.nvdm.p_x, feed_dict={self.nvdm.x: x})
        word_idxs = np.array(cur_ps).argsort()[-sample_size:][::-1]
        ps = cur_ps[word_idxs]

        for idx, (cur_p, word_idx) in enumerate(zip(ps, word_idxs)):
          print("  [%d] %-20s: %.8f" % (idx+1, rev_vocab[word_idx], cur_p))
          p *= cur_p

          print(" [*] perp : %8.f" % -np.log(p))

    def load(self, checkpoint_dir):
        # self.saver = tf.train.Saver()

        print(" [*] Loading checkpoints...")
        model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
          print("Loaded earlier model.")
          return True
        else:
          print("Loading failed. Initializing new parameters.")
          return False

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






if __name__=="__main__":

    #0 for train 1 for evaluation 2 for prediction
    mode = 0
    en_vocab_size = 80000
    batch_size = 25
    dim=200
    vocab_path = "data/vocab_80k.en"
    data_path = "ten_million_english.txt"
    model_dir = "model_data/"
    parser = argparse.ArgumentParser( description = 'Neural Document Modeling')


    parser.add_argument('--vocab_path', type=str, default=vocab_path,
                        help='Path for vocab. (Defaults to %s)' % (vocab_path,))
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='Matching  filename. (Defaults to %s)' % (data_path,))
    parser.add_argument('--model_dir', type=str, default=model_dir,
                        help='Matching  Model directory. (Defaults to %s)' % (model_dir,))
    parser.add_argument('--dim', type=int, default=dim, help='dimensionality for text . (Defaults to %s)' % (dim,))
    parser.add_argument('--vocab', type=int, default=en_vocab_size, help='Location for texts . (Defaults to %s)' % (en_vocab_size,))
    args = parser.parse_args()

    # limit = args.limit
    vocab_path = args.vocab_path
    data_path = args.data_path
    model_dir= args.model_dir
    dim = args.dim
    en_vocab_size = args.vocab

    variational_dm = Neural_DM(batch_size=batch_size, model_dir=model_dir, latent_dim=dim)
    # data = generate(data_path)
    data = generate_simple(data_path)
    create_vocabulary_data(vocab_path, data, en_vocab_size)
    train_set =read_data(data,vocab_path)
    vocab, rev_voc = initialize_vocabulary(vocab_path)
    out = variational_dm.train( train_set, en_vocab_size, vocab, rev_voc )
    print out