from tensorflow.python.ops.init_ops import RandomUniform

from guesswhat.models.abstract_model import *
from guesswhat.models import rnn, utils, resnet, attention
from guesswhat.models.conditional_bn import ConditionalBatchNorm


# 0.5217136
class VQANetwork(ResnetModel):
    def __init__(self, config, no_words, no_answers, image_input, reuse=False, device=''):
        AbstractModel.__init__(self, "vqa", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope_name:


            self.batch_size = None

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer_count = tf.placeholder(tf.float32, [self.batch_size, no_answers], name='answer_count')

            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep),
                                   lambda: tf.constant(1.0))

            word_emb = utils.get_embedding(self._question,
                                           n_words=no_words,
                                           n_dim=int(config["word_embedding_dim"]),
                                           scope="word_embedding")

            if 'glove' in config and config['glove']:
                print("using GLOVE")
                embed_b = tf.get_variable("biases", [int(config["word_embedding_dim"])])
                word_emb = tf.nn.tanh(word_emb + embed_b)
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb = tf.concat([word_emb, self._glove], axis=2)



            # word_emb = tf.nn.dropout(word_emb, dropout_keep)

            self.question_lstm, self.all_lstm_states = rnn.variable_length_LSTM(
                word_emb,
                num_hidden=int(config["no_hidden_LSTM"]),
                dropout_keep_prob=dropout_keep,
                seq_length=self._seq_length,
                depth=int(config["no_LSTM_cell"]),
                scope="question_lstm",
                reuse=reuse)


            #####################
            #   PICTURES
            #####################

            if image_input == "fc8" \
                    or image_input == "fc7" \
                    or image_input == "dummy":

                self._picture = tf.placeholder(tf.float32, [self.batch_size, config["fc_dim"]], name='picture')

                self.picture_out = self._picture
                if config["normalize"]:
                    self.picture_out = tf.nn.l2_normalize(self._picture, dim=1, name="fc_normalization")

            elif image_input.startswith("conv") or image_input == "raw":

                self._picture = tf.placeholder(tf.float32, [self.batch_size,
                                                              config['image']['height'],
                                                              config['image']['width'],
                                                              config['image']['channel']],
                                                 name='picture')

                if image_input == "raw":
                    cbn = None
                    if config["cbn"]["use_cbn"]:

                        cbn_mode=config["cbn"]["mode"]
                        if cbn_mode == "LSTM":
                            shared = config['cbn'].get("shared", False)
                            cbn_network_builder = resnet.CBNfromLSTM(self.question_lstm, config['cbn'],
                                                                     use_shared=shared)

                        elif cbn_mode == "LSTMandFM":
                            cbn_network_builder = resnet.CBNfromLSTMandFM(self.question_lstm, dropout_keep, config["cbn"])
                        elif cbn_mode == "LSTMandAvgFM":
                            cbn_network_builder = resnet.CBNfromLSTMandAvgFM(self.question_lstm, config["cbn"])
                        elif cbn_mode == "CBNfromFMtoAccLSTM":
                            cbn_network_builder = resnet.CBNfromFMtoAccLSTM(self.all_lstm_states, dropout_keep, config["cbn"])
                        elif cbn_mode == "CBNfromLSTMandFM_3":
                            cbn_network_builder = resnet.CBNfromLSTMandFM_3(self.question_lstm, dropout_keep, config["cbn"])
                        elif cbn_mode == "UnrolledLSTM":
                            self.seq_mask = tf.placeholder(tf.float32, [None, None], name="seq_mask")
                            cbn_network_builder = resnet.CBNfromUnrolledLSTM(self.all_lstm_states, self.seq_mask, config["cbn"])
                        else:
                            assert False

                        excluded_scopes = []
                        if 'excluded_scope_names' in config:
                            excluded_scopes = config['excluded_scope_names']

                        apply_bn_before_cbn = False
                        if 'apply_bn_before_cbn' in config['cbn']:
                            apply_bn_before_cbn = config["cbn"]['apply_bn_before_cbn']

                        cbn = ConditionalBatchNorm(cbn_network_builder, excluded_scope_names=excluded_scopes,
                                                                        is_training=self._is_training,
                                                                        apply_bn_before_cbn=apply_bn_before_cbn)
                    resnet_version = 50
                    if 'resnet_version' in config:
                        resnet_version = config['resnet_version']

                    picture_feature_maps = resnet.create_resnet(self._picture,
                                                                is_training=self._is_training,
                                                                scope=scope_name.name,
                                                                cbn=cbn,
                                                                resnet_version=resnet_version)

                    self.picture_feature_maps = picture_feature_maps
                    if config.get('normalize_conv_feat', False):
                        self.picture_feature_maps = tf.nn.l2_normalize(self.picture_feature_maps, dim=[1, 2, 3])
                else:
                    picture_feature_maps = self._picture

                attention_mode = config['attention']["mode"]
                if attention_mode == "none":
                    self.picture_out = tf.reduce_mean(picture_feature_maps, axis=(1, 2))
                elif attention_mode == "classic":
                    print("classic")
                    self.picture_out = attention.compute_attention(picture_feature_maps,
                                                                   self.question_lstm,
                                                                   no_mlp_units=config['attention']['no_attention_mlp'])
                elif attention_mode == "glimpse":
                    self.picture_out = attention.compute_glimpse(picture_feature_maps,
                                                                 self.question_lstm,
                                                                 no_glims=config['attention']['no_glimpses'],
                                                                 glimse_embedding_size=config['attention']['no_attention_mlp'],
                                                                 keep_dropout=dropout_keep)
                else:
                    assert False
            else:
                assert False

            #####################
            #   COMBINE
            #####################
            activation_name = config["activation"]
            with tf.variable_scope('final_mlp'):

                self.question_embedding = utils.fully_connected(self.question_lstm, config["no_question_mlp"], activation=activation_name, scope='question_mlp')
                self.picture_embedding = utils.fully_connected(self.picture_out, config["no_picture_mlp"], activation=activation_name, scope='picture_mlp')

                full_embedding = self.picture_embedding * self.question_embedding
                full_embedding = tf.nn.dropout(full_embedding, dropout_keep)

                out = utils.fully_connected(full_embedding, config["no_hidden_final_mlp"], scope='layer1', activation=activation_name)
                out = tf.nn.dropout(out, dropout_keep)
                out = utils.fully_connected(out, no_answers, activation='linear', scope='layer2')

            # improve soft loss
            answer_count = tf.minimum(self._answer_count, 3)
            #answer_count = tf.where(tf.greater(self._answer_count, 2),
            #                        self._answer_count,
            #                        tf.zeros_like(self._answer_count)) # cf MUTAN loss


            normalizing_sum = tf.maximum(1.0, tf.reduce_sum(answer_count, 1, keep_dims=True))
            self.answer_prob = answer_count / normalizing_sum
            self.soft_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=self.answer_prob, name='soft_cross_entropy')
            self.soft_loss = self.soft_cross_entropy



            self.target_answer = tf.argmax(self._answer_count, axis=1)
            # unmorm_log_prob = tf.log(self._answer_count)
            # self.target_answer = tf.multinomial(unmorm_log_prob, num_samples=1)
            # self.target_answer = tf.reshape(self.target_answer, shape=[-1])

            self.hard_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=self.target_answer, name='hard_cross_entropy')
            self.hard_loss = self.hard_cross_entropy

            if config['loss'] == 'soft':
                self.loss = self.soft_loss
            else:
                self.loss = self.hard_loss

            self.loss = tf.reduce_mean(self.loss)
            self.softmax = tf.nn.softmax(out, name='answer_prob')
            self.prediction = tf.argmax(out, axis=1, name='predicted_answer')  # no need to compute the softmax

            with tf.variable_scope('accuracy'):
                ind = tf.range(tf.shape(self.prediction)[0]) * no_answers + tf.cast(self.prediction, tf.int32)
                pred_count = tf.gather(tf.reshape(self._answer_count, [-1]), ind)
                self.extended_accuracy = tf.minimum(pred_count / 3.0, 1.0, name="extended_accuracy")
                self.accuracy = tf.reduce_mean(self.extended_accuracy)

            tf.summary.scalar('soft_loss', self.soft_loss)
            tf.summary.scalar('hard_loss', self.hard_loss)
            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')


if __name__ == '__main__':
    num_words = 1000
    num_answer = 200
    image_input = "fc8"
    config_model = {
        "fc8_dim": 2048,
        "word_embedding_dim": 256,
        "no_hidden_LSTM": 1024,
        "no_hidden_fc8_mlp": 1024,
        "no_hidden_final_mlp": 1024,
        "no_answer": 1000
    }

    network = VQANetwork(config_model, no_words=num_words, image_input=image_input)
