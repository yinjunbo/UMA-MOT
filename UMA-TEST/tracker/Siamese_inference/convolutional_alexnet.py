import logging
import tensorflow as tf
from tracker.Siamese_utils.misc_utils import get
import config.config as CONFIG

slim = tf.contrib.slim


def convolutional_alexnet_arg_scope(embed_config,
                                    trainable=True,
                                    is_training=False):
  """Defines the default arg scope.

  Args:
    embed_config: A dictionary which contains configurations for the embedding function.
    trainable: If the weights in the embedding function is trainable.
    is_training: If the embedding function is built for training.

  Returns:
    An `arg_scope` to use for the convolutional_alexnet models.
  """
  # Only consider the model to be in training mode if it's trainable.
  # This is vital for batch_norm since moving_mean and moving_variance
  # will get updated even if not trainable.
  is_model_training = trainable and is_training

  if get(embed_config, 'use_bn', True):
    batch_norm_scale = get(embed_config, 'bn_scale', True)
    batch_norm_decay = 1 - get(embed_config, 'bn_momentum', 3e-4)
    batch_norm_epsilon = get(embed_config, 'bn_epsilon', 1e-6)
    batch_norm_params = {
      "scale": batch_norm_scale,
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # Epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "trainable": trainable,
      "is_training": is_model_training,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
    normalizer_fn = slim.batch_norm
  else:
    batch_norm_params = {}
    normalizer_fn = None

  weight_decay = get(embed_config, 'weight_decay', 5e-4)
  if trainable:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  init_method = get(embed_config, 'init_method', 'kaiming_normal')
  if is_model_training:
    logging.info('embedding init method -- {}'.format(init_method))
  if init_method == 'kaiming_normal':
    # The same setting as siamese-fc
    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,
      padding='VALID',
      trainable=trainable,
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.batch_norm], is_training=True) as arg_sc:
        return arg_sc


def convolutional_alexnet(inputs, stage='init', reuse=None, scope='convolutional_alexnet'):
  """Defines the feature extractor of SiamFC.

  Args:
    inputs: a Tensor of shape [batch, h, w, c].
    reuse: if the weights in the embedding function are reused.
    scope: the variable scope of the computational graph.

  Returns:
    net: the computed features of the inputs.
    end_points: the intermediate outputs of the embedding function.
  """

  with tf.variable_scope(scope, 'convolutional_alexnet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      net = inputs
      net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')

      net = slim.conv2d(net, 256, [5, 5], 1, scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], 1, scope='conv4')

      track_feature = slim.conv2d(net, 256, [3, 3], 1, activation_fn=None, normalizer_fn=None, scope='conv5_track')
      reid_feature = slim.conv2d(net, 256, [3, 3], 1, scope='conv5_reid')

      with tf.name_scope('attention'):
        def attach_attention_module(net, attention_module, block_scope=None,
                                    reuse=False):
          def se_block(input_feature, name, reuse=False, ratio=4):
            """Contains the implementation of Squeeze-and-Excitation(SE) block.
            As described in https://arxiv.org/abs/1709.01507.
            """

            kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
            bias_initializer = tf.constant_initializer(value=0.0)

            with tf.variable_scope(name):
              channel = input_feature.get_shape()[-1]
              # Global average pooling
              squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)
              assert squeeze.get_shape()[1:] == (1, 1, channel)
              excitation = tf.layers.dense(inputs=squeeze,
                                           units=channel // ratio,
                                           activation=tf.nn.relu,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                           name='bottleneck_fc',
                                           reuse=reuse)
              assert excitation.get_shape()[1:] == (1, 1, channel // ratio)
              excitation = tf.layers.dense(inputs=excitation,
                                           units=channel,
                                           activation=tf.nn.sigmoid,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                           name='recover_fc',
                                           reuse=reuse)
              assert excitation.get_shape()[1:] == (1, 1, channel)
              scale = input_feature * excitation
              return scale

          if attention_module == 'se_block':  # SE_block
            block_scope = 'se_block' if block_scope is None else block_scope + '_SE'
            net = se_block(net, block_scope, reuse)
          else:
            raise Exception("'{}' is not supported attention module!".format(attention_module))
          return net

        if CONFIG.ATTENTION:
          track_feature = attach_attention_module(track_feature, CONFIG.ATTENTION, 'tracking', reuse=reuse)
          reid_feature = attach_attention_module(reid_feature, CONFIG.ATTENTION, 're-id', reuse=reuse)
        if stage == 'init':
          reid_feature = tf.reduce_mean(reid_feature, axis=[1, 2], keep_dims=True)  # GAP
          reid_feature = tf.squeeze(reid_feature, [1, 2], name='embedding/squeezed')


        return track_feature, reid_feature


convolutional_alexnet.stride = 8
