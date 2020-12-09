# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""
Copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py
This is intend to implemente the MXNet insightface version ResNet

Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
import tensorflow as tf

BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101':
        ('34fb605428fcc7aa4d62f44404c11509', '0f678c91647380debd923963594981b3')
}

layers = None


def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation='softmax',
           **kwargs):
  global layers
  if 'layers' in kwargs:
    layers = kwargs.pop('layers')
  else:
    layers = VersionAwareLayers()
  if kwargs:
    raise ValueError('Unknown argument(s): %s' % (kwargs,))
  if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  x = layers.ZeroPadding2D(padding=1, name='conv1_pad')(img_input)
  x = layers.Conv2D(64, 3, strides=1, use_bias=use_bias, kernel_initializer='glorot_normal', name='conv1_conv')(x)

  if not preact:
    x = layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name='conv1_bn')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='conv1_prelu')(x)

  # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
  # x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

  x = stack_fn(x)

  if preact:
    x = layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name='post_bn')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='post_prelu')(x)

  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name=model_name)

  # Load weights.
  if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
    if include_top:
      file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
      file_hash = WEIGHTS_HASHES[model_name][0]
    else:
      file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
      file_hash = WEIGHTS_HASHES[model_name][1]
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False, kernel_initializer='glorot_normal', name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_1_bn')(x)
  x = layers.ZeroPadding2D(padding=1, name=name + '_1_pad')(x)
  x = layers.Conv2D(filters, 3, strides=1, kernel_initializer='glorot_normal', use_bias=False, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_2_bn')(x)
  x = layers.PReLU(shared_axes=[1, 2], name=name + '_1_prelu')(x)

  x = layers.ZeroPadding2D(padding=1, name=name + '_2_pad')(x)
  x = layers.Conv2D(filters, kernel_size, strides=stride, kernel_initializer='glorot_normal', use_bias=False, name=name + '_2_conv')(x)
  x = layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  return x


def stack1(x, filters, blocks, stride1=2, name=None):
  x = block1(x, filters, stride=stride1, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
  return x

def ResNet34(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
  """Instantiates the ResNet34 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, False, 'resnet34', include_top, weights,
                input_tensor, input_shape, pooling, classes, **kwargs)

def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
  """Instantiates the ResNet50 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 14, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, False, 'resnet50', include_top, weights,
                input_tensor, input_shape, pooling, classes, **kwargs)

def ResNet100(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
  """Instantiates the ResNet50 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, name='conv2')
    x = stack1(x, 128, 13, name='conv3')
    x = stack1(x, 256, 30, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, False, 'resnet100', include_top, weights,
                input_tensor, input_shape, pooling, classes, **kwargs)

def ResNet101(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
  """Instantiates the ResNet50 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 23, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, False, 'resnet101', include_top, weights,
                input_tensor, input_shape, pooling, classes, **kwargs)
