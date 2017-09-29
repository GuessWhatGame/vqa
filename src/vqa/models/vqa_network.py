import tensorflow as tf


from generic.tf_models import rnn, utils
from vqa.models.resnet_model import ResnetModel
from generic.tf_models.image_feature import get_image_features

class VQANetwork(ResnetModel):
    def __init__(self, config, no_words, no_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "vqa", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:


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
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb = tf.concat([word_emb, self._glove], axis=2)


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

            self._picture = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='picture')
            self.picture_out = get_image_features(
                    image=self._picture, question=self.question_lstm,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['image']
                )


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

    def get_loss(self):
        return self.loss



