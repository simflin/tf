"""
Implementation of ENet architecture (https://arxiv.org/pdf/1606.02147.pdf) for semantic
segmentation using Tensorflow (ver >=1.0).

See also https://github.com/e-lab/ENet-training - original Torch implementation
"""

import scipy as sp
import scipy.misc

import numpy as np
import tensorflow as tf

user_ops_pooling = tf.load_op_library("unpooling.so")
unpooling = user_ops_pooling.unpooling

from tensorflow.python.framework import ops
BN_ALPHA = 0.9


@ops.RegisterShape("Unpooling")
def _unpooling_shape(op):
    input_shape_shape = op.inputs[2].get_shape().with_rank(1)
    return [[None, None, None, op.inputs[1].get_shape()[-1].value]]


def prelu(x, name):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - tf.abs(x)) * 0.5

    return pos + neg


def initial_block(x, is_training):
    with tf.variable_scope('initial_block'):
        initial1 = tf.layers.max_pooling2d(x, pool_size=(2,2), strides=2, padding='same',
                                            name='pool')
        initial2 = tf.layers.conv2d(x, filters=13, kernel_size=3, padding='same', strides=2,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                activation=None, name="conv")
        x = tf.concat([initial1, initial2], 3)
        x = tf.layers.batch_normalization(x, momentum=BN_ALPHA, training=is_training, name="bn")
        x = prelu(x, "prelu")
        return x


def bottleneck(x, n_filters, internal_scale, downsample, t, is_training, name, dropout_prob,
                dilation_rate=None):
    with tf.variable_scope(name):
        projection_size = n_filters / internal_scale
        kern = 2 if downsample else 1

        projection = tf.layers.conv2d(x, filters=projection_size, kernel_size=kern, padding='same',
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                strides=kern, activation=None, name="conv_projection", use_bias=False)
        projection = tf.layers.batch_normalization(projection, momentum=BN_ALPHA,
                training=is_training, name="projection_bn")
        projection = prelu(projection, 'projection_relu')
        if t == 'dilated':
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=3,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    padding='same', dilation_rate=dilation_rate, activation=None, name='dilation')
        elif t == 'asymetric':
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=(5,1),
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    padding='same', activation=None, name='asymmetric_5x1', use_bias=False)
            conved = tf.layers.conv2d(conved, filters=projection_size, kernel_size=(1,5),
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    padding='same', activation=None, name='asymmetric_1x5')
        elif t == 'regular':
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=3,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    padding='same', strides=1, activation=None, name="conv")
        elif t == 'deconv':
            conved = tf.layers.conv2d_transpose(projection, filters=projection_size, kernel_size=2,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    activation=None, strides=2, padding='same', name='deconv')
        else:
            assert False, name + " " + t
        conved = tf.layers.batch_normalization(conved,momentum=BN_ALPHA, training=is_training,
                                               name="main_conv_bn")
        conved = prelu(conved, "main_conv_prelu")
        expansion = tf.layers.conv2d(conved, filters=n_filters, kernel_size=1, padding='same',
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                strides=1, activation=None, name="conv_expansion", use_bias=False)
        expansion = tf.layers.batch_normalization(expansion, momentum=BN_ALPHA,
                                                  training=is_training, name="expansion_bn")
        s = tf.shape(expansion)
        s = tf.slice(s, [0], [1])
        s = tf.concat([s, [1,1,n_filters]], 0)
        if is_training:
            expansion = tf.nn.dropout(expansion, keep_prob=dropout_prob, noise_shape=s)

        if downsample:
            shape = tf.shape(x)
            y, indices = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1],
                    strides=[1,2,2,1], padding='SAME', name='pool', Targmax=tf.int64)
            output = np.array(y.get_shape())[-1].value
            y = tf.pad(y, [[0,0], [0,0], [0,0], [0, n_filters-output]])
        else:
            y = x
        x = expansion + y
        x = prelu(x, 'final_prelu')
    if downsample:
        return x, shape, indices
    else:
        return x


def bottleneck_decoder(x, n_filters, internal_scale, upsample, is_training, name,
                        mp_indices=None, mp_shape=None):
    with tf.variable_scope(name):
        projection_size = n_filters / internal_scale
        projection = tf.layers.conv2d(x, filters=projection_size, kernel_size=1, padding='same',
                strides=1, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                activation=None, name="conv_projection", use_bias=False)
        projection = tf.layers.batch_normalization(projection, momentum=BN_ALPHA,
                training=is_training, name="projection_bn")
        projection = tf.nn.relu(projection)  # FIXME prelu ?

        if upsample:
            conved = tf.layers.conv2d_transpose(projection, filters=projection_size, kernel_size=3,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    padding='same', strides=2, activation=None, name='dilation')
        else:
            conved = tf.layers.conv2d(projection, filters=projection_size, kernel_size=3,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    padding='same', strides=1, activation=None, name="conv")
        conved = tf.layers.batch_normalization(conved, momentum=BN_ALPHA, training=is_training,
                name="main_conv_bn")
        conved = tf.nn.relu(conved)  # FIXME prelu?

        expansion = tf.layers.conv2d(conved, filters=n_filters, kernel_size=1, padding='same',
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                strides=1, activation=None, name="conv_expansion", use_bias=False)
        expansion = tf.layers.batch_normalization(expansion, momentum=BN_ALPHA,
                training=is_training, name="expansion_bn")

        if upsample:
            y = tf.layers.conv2d(x, filters=n_filters, kernel_size=1, padding='same',
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    strides=1, activation=None, name="identity_projection", use_bias=False)
            y = tf.layers.batch_normalization(y, training=is_training, momentum=BN_ALPHA,
                    name="identity_projection_bn")
            y = unpooling(tf.cast(mp_indices, tf.int32), y, mp_shape)
            # conv2d_transpose increases space dims of batch with factor 2 while unpooling has
            # output size as its parameter (and it is the 'correct' output size since it is the
            # size of maxpooling input tensor). So we need to crop 'expansion' in order to match
            # unpooling output size
            expansion = tf.slice(expansion, [0,0,0,0], tf.shape(y))
        else:
            y = x

        x = expansion + y
        x = tf.nn.relu(x) # FIXME
    return x


def build_graph(input_batch, is_training, num_classes=2):
    # x - 4d tensor
    x = initial_block(input_batch, is_training)

    x, mp_shape1, mp_indices1 = bottleneck(x, n_filters=64, internal_scale=4, downsample=True,
            t='regular', is_training=is_training, name="bottleneck1.0", dropout_prob=0.01)
    for i in xrange(1,5):
        x = bottleneck(x, n_filters=64, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck1.{}".format(i), dropout_prob=0.01)

    x, mp_shape2, mp_indices2 = bottleneck(x, n_filters=128, internal_scale=4, downsample=True,
            t='regular', is_training=is_training, name="bottleneck2.0", dropout_prob=0.1)
    for j in xrange(2,4):
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck{}.1".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated',
                dilation_rate=2, is_training=is_training, name="bottleneck{}.2".format(j),
                dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='asymetric',
                is_training=is_training, name="bottleneck{}.3".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated',
                dilation_rate=4, is_training=is_training, name="bottleneck{}.4".format(j),
                dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='regular',
                is_training=is_training, name="bottleneck{}.5".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated',
                dilation_rate=8, is_training=is_training, name="bottleneck{}.6".format(j),
                dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='asymetric',
                is_training=is_training, name="bottleneck{}.7".format(j), dropout_prob=0.1)
        x = bottleneck(x, n_filters=128, internal_scale=4, downsample=False, t='dilated',
                dilation_rate=16, is_training=is_training, name="bottleneck{}.8".format(j),
                dropout_prob=0.1)

    x = bottleneck_decoder(x, 64, internal_scale=4, upsample=True, is_training=is_training,
            name='decoder/bottleneck4.0', mp_indices=mp_indices2, mp_shape=mp_shape2)
    for i in xrange(1,3):
        x = bottleneck_decoder(x, 64, internal_scale=4, upsample=False, is_training=is_training,
                name='decoder/bottleneck4.{}'.format(i))

    x = bottleneck_decoder(x, 16, internal_scale=4, upsample=True, is_training=is_training,
            name='decoder/bottleneck5.0', mp_indices=mp_indices1, mp_shape=mp_shape1)
    x = bottleneck_decoder(x, 16, internal_scale=4, upsample=False, is_training=is_training,
            name='decoder/bottleneck5.1')

    logits = tf.layers.conv2d_transpose(x, filters=num_classes, kernel_size=2, strides=2,
            padding='same', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            activation=None, name='conv_decoder_final')

    # crop to input shape
    space_dims = tf.slice(tf.shape(input_batch), [1], [2])
    logits = tf.slice(logits, [0,0,0,0], tf.concat([[-1], space_dims, [-1]], 0))

    return logits


class ENet(object):
    def __init__(self, network_state_filename, is_training=False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.graph_input = tf.placeholder(tf.float32, [None, None, None, 3])
            self.is_training = is_training
            logits = build_graph(self.graph_input, self.is_training)
            self.batch_predictions = tf.nn.softmax(logits)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            NUM_THREADS = 2
            config.intra_op_parallelism_threads=NUM_THREADS
            config.inter_op_parallelism_threads=NUM_THREADS

            saver = tf.train.Saver()
            self.session = tf.Session(config=config)
            saver.restore(self.session, network_state_filename)

    def segment_image(self, img):
        if isinstance(img, str):
            img = sp.misc.imread(img)

        img = img[np.newaxis]  # batch size dimension
        img = img.astype(np.float32) / 255 - 0.5

        [predictions] = self.session.run([self.batch_predictions],
                feed_dict={self.graph_input: img})
        return predictions

