import tensorflow as tf
from generic.tf_models.abstract_network import AbstractNetwork

class ResnetModel(AbstractNetwork):

    def __init__(self, scope_name, device=''):
        super(ResnetModel, self).__init__(scope_name, device)

    def get_parameters(self, finetune=list()):

        params = super(ResnetModel, self).get_parameters()
        params = [ v for v in params if (not 'resnet' in v.name or 'cbn_input' in v.name)]

        if len(finetune) > 0:
            for e in finetune:
                fine_tuned_params = [v for v in tf.trainable_variables() if e in v.name and v not in params]
                params += fine_tuned_params

        return params


    def get_resnet_parameters(self):
        # trainable_vars = [v for v in tf.trainable_variables() if
        #                   ('resnet' in v.name and not 'cbn_input' in v.name)]
        # moving_moments = [v for v in tf.global_variables() if
        #                   'resnet' in v.name and
        #                   'BatchNorm' in v.name
        #                   and not 'Adam' in v.name]
        # return trainable_vars + moving_moments
        return [v for v in tf.global_variables() if self.scope_name in v.name and
            ('resnet' in v.name and
             not 'cbn_input' in v.name and
             not 'Adam' in v.name and
             "local_step" not in v.name and
             "moving_mean/biased" not in v.name)]