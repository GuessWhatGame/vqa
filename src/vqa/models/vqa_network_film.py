import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_factory.fusion_factory import get_fusion_mechanism
import neural_toolbox.rnn as rnn

from generic.tf_factory.image_factory import get_image_features
from generic.tf_utils.abstract_network import ResnetModel

from neural_toolbox.film_stack import FiLM_Stack
from neural_toolbox.reading_unit import create_reading_unit, create_film_layer_with_reading_unit


class VQANetwork_FiLM(ResnetModel):
    def __init__(self, config, no_words, no_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "vqa_film", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):

            self.batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [self.batch_size, no_answers], name='answer')
            self._answer_count = tf.placeholder(tf.float32, [self.batch_size, no_answers], name='answer_count')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=no_words,
                embed_dim=config["question"]["word_embedding_dim"],
                scope="word_embedding",
                reuse=reuse)

            if config["question"]['glove']:
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb = tf.concat([word_emb, self._glove], axis=2)

            word_emb = tf.nn.dropout(word_emb, dropout_keep)
            self.rnn_states, self.last_rnn_states = rnn.gru_factory(
                inputs=word_emb,
                seq_length=self._seq_length,
                num_hidden=config["question"]["rnn_state_size"],
                bidirectional=config["question"]["bidirectional"],
                max_pool=config["question"]["max_pool"],
                layer_norm=config["question"]["layer_norm"],
                reuse=reuse)

            self.last_rnn_states = tf.nn.dropout(self.last_rnn_states, dropout_keep)
            self.rnn_states = tf.nn.dropout(self.rnn_states, dropout_keep)  # Note that the last_states may have a different dropout... TODO: study impact

            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='image')
            with tf.variable_scope("image", reuse=reuse):
                self.image_out = get_image_features(
                    image=self._image, question=self.last_rnn_states,
                    is_training=self._is_training,
                    scope_name="image_processing",
                    config=config['image'],
                    dropout_keep=dropout_keep
                )

            # apply attention or use vgg features
            if len(self.image_out.get_shape()) == 2:
                self.visual_embedding = self.image_out

            else:

                with tf.variable_scope("image_reading_cell"):

                    self.reading_unit = create_reading_unit(last_state=self.last_rnn_states,
                                                            states=self.rnn_states,
                                                            seq_length=self._seq_length,
                                                            keep_dropout=dropout_keep,
                                                            config=config["film_input"]["reading_unit"],
                                                            reuse=reuse)

                    film_layer_fct = create_film_layer_with_reading_unit(self.reading_unit)

                with tf.variable_scope("image_film_stack", reuse=reuse):

                    self.film_img_stack = FiLM_Stack(image=self.image_out,
                                                     film_input=[],
                                                     attention_input=self.last_rnn_states,
                                                     film_layer_fct=film_layer_fct,
                                                     is_training=self._is_training,
                                                     dropout_keep=dropout_keep,
                                                     config=config["film_block"],
                                                     reuse=reuse)

                    film_img_output = self.film_img_stack.get()
                    film_img_output = tf.nn.dropout(film_img_output, dropout_keep)

                    self.visual_embedding = film_img_output

            #####################
            #   FUSION LAYER
            #####################

            with tf.variable_scope('fusion'):

                language_embedding = self.last_rnn_states
                if config["fusion"]["mode"] == "none":
                    language_embedding = None

                self.vqa_embedding = get_fusion_mechanism(input1=self.visual_embedding,
                                                          input2=language_embedding,
                                                          config=config["fusion"],
                                                          dropout_keep=dropout_keep,
                                                          reuse=reuse)

            #####################
            #   FINAL LAYER
            #####################

            with tf.variable_scope("classifier", reuse=reuse):

                if config["classifier"]["no_mlp_units"] > 0:
                    self.hidden_state = tfc_layers.fully_connected(self.vqa_embedding,
                                                                   num_outputs=config["classifier"]["no_mlp_units"],
                                                                   activation_fn=tf.nn.relu,
                                                                   reuse=reuse,
                                                                   scope="classifier_hidden_layer")

                    self.hidden_state = tf.nn.dropout(self.hidden_state, dropout_keep)
                else:
                    self.hidden_state = self.vqa_embedding

                self.out = tfc_layers.fully_connected(self.hidden_state,
                                                      num_outputs=no_answers,
                                                      activation_fn=None,
                                                      reuse=reuse,
                                                      scope="classifier_softmax_layer")

            #####################
            #   Loss
            #####################

                # improve soft loss
                answer_count = tf.minimum(self._answer_count, 3)

                normalizing_sum = tf.maximum(1.0, tf.reduce_sum(answer_count, 1, keep_dims=True))
                self.answer_prob = answer_count / normalizing_sum
                self.soft_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.answer_prob, name='soft_cross_entropy')
                self.soft_loss = self.soft_cross_entropy

                self.target_answer = tf.argmax(self._answer_count, axis=1)
                # unmorm_log_prob = tf.log(self._answer_count)
                # self.target_answer = tf.multinomial(unmorm_log_prob, num_samples=1)
                # self.target_answer = tf.reshape(self.target_answer, shape=[-1])

                self.hard_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.target_answer, name='hard_cross_entropy')
                self.hard_loss = self.hard_cross_entropy

                if config['loss'] == 'soft':
                    self.loss = self.soft_loss
                else:
                    self.loss = self.hard_loss

                self.loss = tf.reduce_mean(self.loss)
                self.softmax = tf.nn.softmax(self.out, name='answer_prob')
                self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

                with tf.variable_scope('accuracy'):
                    ind = tf.range(tf.shape(self.prediction)[0]) * no_answers + tf.cast(self.prediction, tf.int32)
                    pred_count = tf.gather(tf.reshape(self._answer_count, [-1]), ind)
                    self.extended_accuracy = tf.minimum(pred_count / 3.0, 1.0, name="extended_accuracy")
                    self.accuracy = tf.reduce_mean(self.extended_accuracy)

            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy


if __name__ == "__main__":

    import json
    with open("../../../config/vqa/film.json", 'r') as f_config:
        config = json.load(f_config)

        VQANetwork_FiLM(config["model"], no_words=354, no_answers=167)
