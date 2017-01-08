import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

linear = rnn_cell._linear

class VariationalTopicModel(object):
    """
    Neural Variational Model for Document Modeling
    """
    def __init__(self, sess, vocab_size, hidden_encoder_dim, latent_dim, model_dir, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout

        self.x = tf.placeholder(tf.float32, [vocab_size], name="input_x")
        self.x_idx = tf.placeholder(tf.int32, [None], name="input_indices")
        with tf.variable_scope("encoder"):
            self.hidden_encoder = tf.nn.relu( linear(tf.expand_dims(self.x, 0), hidden_encoder_dim, bias=True, scope="lambda"))
            self.pi = tf.nn.relu( linear(self.hidden_encoder, hidden_encoder_dim, bias=True, scope="pi"))

            # Mean  Encoder
            self.mu_encoder = linear(self.pi, latent_dim, bias=True, scope="mu_encoder")

            # Sigma  Encoder
            self.logvar_encoder = linear(self.pi, latent_dim, bias=True, scope="logvar_encoder")

            # Sample epsilon
            self.epsilon = tf.random_normal((1, latent_dim), name='epsilon')

            self.std_dev = tf.sqrt(tf.exp(self.logvar_encoder ))
            self.h = self.mu_encoder + self.std_dev * self.epsilon
            _ = tf.histogram_summary("mu", self.mu_encoder)
            _ = tf.histogram_summary("sigma", self.logvar_encoder)
            _ = tf.histogram_summary("h", self.h)
            _ = tf.histogram_summary("mu + sigma", self.mu_encoder + self.logvar_encoder)

        with tf.variable_scope("decoder"):
            self.R = tf.get_variable("R", [vocab_size, latent_dim])
            self.b = tf.get_variable("b", [vocab_size])

            self.e = -tf.matmul(self.h, self.R, transpose_b=True) + self.b
            self.p_x = tf.squeeze(tf.nn.softmax(self.e))


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)


        # Calculate loss
        with tf.name_scope("loss"):
            # KL Divergence Loss
            self.kl_divergence = -0.5*tf.reduce_sum(1.0 + self.logvar_encoder - tf.square(self.mu_encoder) - tf.exp(self.logvar_encoder))
            # Log likelihood
            self.likelihood = -tf.reduce_sum(tf.log(tf.gather(self.p_x, self.x_idx) + 1e-6))
            self.loss = self.likelihood + self.kl_divergence
            _ = tf.scalar_summary("encoder loss", self.kl_divergence)
            _ = tf.scalar_summary("generator loss", self.likelihood )
            _ = tf.scalar_summary("total loss", self.loss )
        self.merged_sum = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("./logs/%s" % model_dir, sess.graph)
        # # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class VariationalTopicModelBatch(object):
    """
    Neural Variational Model for Document Modeling
    """
    def __init__(self, sess, vocab_size, hidden_encoder_dim, latent_dim, model_dir=None, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout

        self.x = tf.placeholder(tf.float32, [None, vocab_size], name="input_x")
        with tf.variable_scope("encoder"):
            self.hidden_encoder = tf.nn.relu( linear(self.x, hidden_encoder_dim, bias=True, scope="lambda"))
            self.pi = tf.nn.relu( linear(self.hidden_encoder, hidden_encoder_dim, bias=True, scope="pi"))

            # Mean  Encoder
            self.mu_encoder = linear(self.pi, latent_dim, bias=True, scope="mu_encoder")

            # Sigma  Encoder
            self.logvar_encoder = linear(self.pi, latent_dim, bias=True, scope="logvar_encoder")

            # Sample epsilon
            self.epsilon = tf.random_normal((tf.shape(self.logvar_encoder)), 0, 1, name='epsilon')

            self.std_dev = tf.sqrt(tf.exp(self.logvar_encoder ))
            self.h = tf.add(self.mu_encoder , tf.mul(self.std_dev , self.epsilon))

            # _ = tf.histogram_summary("mu", self.mu_encoder)
            # _ = tf.histogram_summary("sigma", self.logvar_encoder)
            # _ = tf.histogram_summary("h", self.h)
            # _ = tf.histogram_summary("mu + sigma", self.mu_encoder + self.logvar_encoder)

        with tf.variable_scope("decoder"):
            self.R = tf.get_variable("R", [vocab_size, latent_dim])
            self.b = tf.get_variable("b", [vocab_size])

            self.e = -tf.matmul(self.h, self.R, transpose_b=True) - self.b
            self.p_x = tf.squeeze(tf.nn.softmax(self.e))


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)


        # Calculate loss
        with tf.name_scope("loss"):
            # KL Divergence Loss
            self.kl_divergence = -0.5*tf.reduce_sum(1.0 + self.logvar_encoder - tf.square(self.mu_encoder) - tf.exp(self.logvar_encoder), 1)
            # Log likelihood
            self.likelihood = -tf.reduce_sum(tf.mul(tf.log(self.p_x+1e-6), self.x), 1)
            self.loss = tf.reduce_mean(self.likelihood + self.kl_divergence)

            _ = tf.scalar_summary("encoder loss", tf.reduce_mean(self.kl_divergence))
            _ = tf.scalar_summary("generator loss", tf.reduce_mean(self.likelihood ))
            _ = tf.scalar_summary("total loss", self.loss )

        if model_dir != None:
            self.merged_sum = tf.merge_all_summaries()
            self.writer = tf.train.SummaryWriter("./logs/%s" % model_dir, sess.graph)
        # # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

