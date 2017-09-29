from tensorflow.python.ops.init_ops import RandomUniform, UniformUnitScaling

import generic.tf_models.utils as utils
import tensorflow as tf
from conditional_batch_norm.conditional_bn import CBNAbtract

# Usefull tools if you want to try funky cbn architecture (We tried something like 5-6 architecture, LSTM is the simplest vs efficiency)
def pooling_to_shape(feature_maps, shape, pooling=tf.nn.avg_pool):
    cur_h = int(feature_maps.get_shape()[1])
    cur_w = int(feature_maps.get_shape()[2])

    if cur_h > shape[0] and cur_w > shape[1]:
        stride_h, stride_w = int(cur_h / shape[0]), int(cur_w / shape[1])
        reduce_fm = pooling(feature_maps, ksize=[1, stride_h, stride_w, 1], strides=[1, stride_h, stride_w, 1], padding="VALID")
    else:
        reduce_fm = feature_maps

    return reduce_fm


class CBNfromLSTM(CBNAbtract):
    def __init__(self, lstm_state, config, use_betas=True, use_gammas=True):
        self.lstm_state = lstm_state
        self.cbn_embedding_size = config['cbn_embedding_size']
        self.use_betas = use_betas
        self.use_gammas = use_gammas


    def create_cbn_input(self, feature_maps):
        no_features = int(feature_maps.get_shape()[3])
        batch_size = tf.shape(feature_maps)[0]

        if self.use_betas:
            h_betas = utils.fully_connected(self.lstm_state,
                                            self.cbn_embedding_size,
                                            weight_initializer=RandomUniform(-1e-4, 1e-4),
                                            scope="hidden_betas",
                                            activation='relu')
            delta_betas = utils.fully_connected(h_betas, no_features, scope="delta_beta",
                                                weight_initializer=RandomUniform(-1e-4, 1e-4), use_bias=False)
        else:
            delta_betas = tf.tile(tf.constant(0.0, shape=[1, no_features]), tf.stack([batch_size, 1]))

        if self.use_gammas:
            h_gammas = utils.fully_connected(self.lstm_state,
                                             self.cbn_embedding_size,
                                             weight_initializer=RandomUniform(-1e-4, 1e-4),
                                             scope="hidden_gammas",
                                             activation='relu')
            delta_gammas = utils.fully_connected(h_gammas, no_features, scope="delta_gamma",
                                                 weight_initializer=RandomUniform(-1e-4, 1e-4))
        else:
            delta_gammas = tf.tile(tf.constant(0.0, shape=[1, no_features]), tf.stack([batch_size, 1]))

        return delta_betas, delta_gammas

