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
"""Various learning rate decay functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.cyclic_lr")
def cyclic_lr(learning_rate, global_step,
                      step_size,
                      base_lr,
                      max_lr,
                      name=None):
  """Applies exponential decay to the learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `global_step / decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Example: decay every 100000 steps with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             100000, 0.96, staircase=True)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate.
    staircase: Boolean.  If `True` decay the learning rate at discrete intervals
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.
  """
  if global_step is None:
    raise ValueError("global_step is required for exponential_decay.")
  with ops.name_scope(
      name, "CyclicLR",
      [learning_rate, global_step, step_size, base_lr, max_lr]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    step_size = math_ops.cast(step_size, dtype)
    base_lr = math_ops.cast(base_lr, dtype)
    max_lr = math_ops.cast(max_lr, dtype)
    
    scale_fn = lambda x: 1/(2.**(x-1))
    cycle = math_ops.floor(1 + global_step/(2 * step_size))
    x = math_ops.abs(global_step/step_size - 2 * cycle + 1)
    dummy = 0
    dummy = math_ops.to_float(dummy)
    lr = base_lr + (max_lr - base_lr) * tf.maximum(dummy, (1-x)) * scale_fn(cycle)
    
    return lr
  

