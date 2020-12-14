# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Code defining LEO inner loop.

See "Meta-Learning with Latent Embedding Optimization" by Rusu et al.
(https://arxiv.org/pdf/1807.05960.pdf).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import data as data_module
import wandb


def get_orthogonality_regularizer(orthogonality_penalty_weight):
  """Returns the orthogonality regularizer."""
  def orthogonality(weight):
    """Calculates the layer-wise penalty encouraging orthogonality."""
    with tf.name_scope(None, "orthogonality", [weight]) as name:
      w2 = tf.matmul(weight, weight, transpose_b=True)
      wn = tf.norm(weight, ord=2, axis=1, keepdims=True) + 1e-32
      correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
      matrix_size = correlation_matrix.get_shape().as_list()[0]
      base_dtype = weight.dtype.base_dtype
      identity = tf.eye(matrix_size, dtype=base_dtype)
      weight_corr = tf.reduce_mean(
          tf.squared_difference(correlation_matrix, identity))
      return tf.multiply(
          tf.cast(orthogonality_penalty_weight, base_dtype),
          weight_corr,
          name=name)

  return orthogonality


class LEO(snt.AbstractModule):
  """Sonnet module implementing the inner loop of LEO."""

  def __init__(self, config=None, use_64bits_dtype=True, name="leo"):
    super(LEO, self).__init__(name=name)
    self.n_splits = 8

    self._float_dtype = tf.float64 if use_64bits_dtype else tf.float32
    self._int_dtype = tf.int64 if use_64bits_dtype else tf.int32

    self._inner_unroll_length = config["inner_unroll_length"]
    self._finetuning_unroll_length = config["finetuning_unroll_length"]
    self._inner_lr_init = config["inner_lr_init"]
    self._finetuning_lr_init = config["finetuning_lr_init"]
    self._num_latents = config["num_latents"]
    self._dropout_rate = config["dropout_rate"]

    self._kl_weight = config["kl_weight"]  # beta
    self._encoder_penalty_weight = config["encoder_penalty_weight"]  # gamma
    self._l2_penalty_weight = config["l2_penalty_weight"]  # lambda_1
    # lambda_2
    self._orthogonality_penalty_weight = config["orthogonality_penalty_weight"]

    assert self._inner_unroll_length > 0, ("Positive unroll length is necessary"
                                           " to create the graph")

  def _build(self, data, is_meta_training=True):
    """Connects the LEO module to the graph, creating the variables.

    Args:
      data: A data_module.ProblemInstance constaining Tensors with the
          following shapes:
          - tr_input: (N, K, dim)
          - tr_output: (N, K, 1)
          - tr_info: (N, K)
          - val_input: (N, K_valid, dim)
          - val_output: (N, K_valid, 1)
          - val_info: (N, K_valid)
            where N is the number of classes (as in N-way) and K and the and
            K_valid are numbers of training and validation examples within a
            problem instance correspondingly (as in K-shot), and dim is the
            dimensionality of the embedding.
      is_meta_training: A boolean describing whether we run in the training
        mode.

    Returns:
      Tensor with the inner validation loss of LEO (include both adaptation in
      the latent space and finetuning).
    """
    if isinstance(data, list):
      data = data_module.ProblemInstance(*data)
    self.is_meta_training = is_meta_training
    self.save_problem_instance_stats(data.tr_input)

    latents, kl, kl_components, kl_zn, distribution_params = self.forward_encoder(data)
    tr_loss, adapted_classifier_weights, encoder_penalty, corr_penalty, adapted_latents, adapted_kl, adapted_kl_components, adapted_kl_zn, spurious = self.leo_inner_loop(
        data, latents, distribution_params)

    val_loss, val_accuracy = self.finetuning_inner_loop(
        data, tr_loss, adapted_classifier_weights)

    #tr_loss can we observe this for each latent component
    #val_loss can we observe this for each latent component
    #compute generalization_loss = val_loss - tr_loss
    #if generalization_loss is high fir a latent component, simply threshold and drop it.
    # graph the generalization loss for the components during the training, are there any that have a high genrealizatio loss

    #remove correlations between latent space gradient dimensions
    val_loss += self._kl_weight * kl
    val_loss += self._encoder_penalty_weight * encoder_penalty
    # The l2 regularization is is already added to the graph when constructing
    # the snt.Linear modules. We pass the orthogonality regularizer separately,
    # because it is not used in self.grads_and_vars.
    regularization_penalty = (
        self._l2_regularization + self._decoder_orthogonality_reg)

    batch_val_loss = tf.reduce_mean(val_loss)
    batch_val_accuracy = tf.reduce_mean(val_accuracy)
    batch_generalization_loss = tf.reshape(tf.reduce_mean(val_loss, 1), [5,1]) - tr_loss

    if self.is_meta_training:
        tr_out = tf.cast(data.tr_output, dtype=tf.float32)
        tr_out_tiled = tf.tile(tr_out, multiples=[1, 1, 64])
        tr_out_tiled_expanded = tf.expand_dims(tr_out_tiled, -1)
        kl_components_y = tf.concat([tr_out_tiled_expanded, tf.expand_dims(kl_components, -1)], axis=-1)
        adapted_kl_components_y = tf.concat([tr_out_tiled_expanded, tf.expand_dims(adapted_kl_components, -1)], axis=-1)
        kl_zn_y = tf.concat([tf.squeeze(tr_out, -1), kl_zn], axis=-1)
        adapted_kl_zn_y = tf.concat([tf.squeeze(tr_out, -1), adapted_kl_zn], axis=-1)
        latents_y = tf.concat([tr_out_tiled_expanded, tf.expand_dims(latents, -1)], axis=-1)
        adapted_latents_y = tf.concat([tr_out_tiled_expanded, tf.expand_dims(adapted_latents, -1)], axis=-1)
        spurious_y = tf.concat([tr_out_tiled_expanded, tf.expand_dims(spurious, -1)], axis=-1)
    else:
        kl_components_y = kl_components
        adapted_kl_components_y = adapted_kl_components
        kl_zn_y = kl_zn
        adapted_kl_zn_y = adapted_kl_zn
        latents_y = latents
        adapted_latents_y = adapted_latents
        spurious_y = spurious
    return batch_val_loss + regularization_penalty, batch_val_accuracy, batch_generalization_loss, \
           kl_components_y, adapted_kl_components_y, kl_zn_y, adapted_kl_zn_y, kl, adapted_kl, latents_y, adapted_latents_y, spurious_y

  # def l2_regularizer(self, phi, theta):
  #     reg = tf.reduce_sum(tf.square(phi - theta))
  #     return 0*5 * 1e-8 * reg

  @snt.reuse_variables
  def leo_inner_loop(self, data, latents, distribution_params):
    with tf.variable_scope("leo_inner"):
      inner_lr = tf.get_variable(
          "lr", [1, 1, self._num_latents],
          dtype=self._float_dtype,
          initializer=tf.constant_initializer(self._inner_lr_init))
    starting_latents = latents
    # learn a function which learns the importance of the splits

    starting_latents_split = tf.nn.l2_normalize(starting_latents, axis=0)#self.get_split_features_full(starting_latents, "l2n")
    # input = tf.zeros([5])
    # for i in range(8): #self._inner_unroll_length):
    #     # adjusted with l2 norm
    #     loss, theta_i = self.forward_decoder(
    #         data, starting_latents_split[i], tf.zeros([5, 1, 640], dtype=tf.float32))
    #     # #idea: if the loss is high when the dimensions are regularized => the input is being correlated with unwanted features (spurious correlations)
    #     if (i==0):
    #         prev_loss_max = loss
    #
    #     mask = tf.greater(loss, prev_loss_max)
    #     mask = tf.squeeze(mask)
    #     input = tf.where(mask, tf.fill([5,], i), tf.cast(input, tf.int32))
    #     prev_loss_max = tf.math.maximum(prev_loss_max, loss)

          #controlling the rate and direction of the latents
          #1. Try l2 on weights /z
          #2 Try divergence on weights /z
          #3 try MSE on weights /z
          # lambda/2||phi - theta||^2 minimize the distance
          # - lambda/2||phi - theta||^2 maxmize  the distance

          # adjustments in the latent space?
          # how can we identify spurious correlations?
          # 1. split the data into predefined knowledge X = (X_1, X_2, X_3...X_8)
          # 2. sample from Z n times Xn = (X_n1, X_n2, X_n3...X_n8)
          # 3. compute the mean for each datum
          # 4. compute the correlation between  Xn

          # 5. learning F s.t.: if we remove this pretrained knowledge, what is the probability the classifier will remain invariant to the change
          # if self.is_meta_training:
          #     # encourages the decoder to output a parameter initizalization which is closest to the adapted latents
          #     corr_penalty = tf.losses.mean_squared_error(
          #         labels=latents, predictions=starting_latents)
          #     corr_penalty = tf.cast(corr_penalty, self._float_dtype)
          # else:
          # loss -= 0.01 * corr_penalty
          # loss -= 0.01*tf.nn.l2_loss((latents-starting_latents))
          # _, adapted_kl, _, _ = self.kl_divergence_given_latents(distribution_params, latents)
          # divergence_penalty = 0.0001 * adapted_kl
          # loss += divergence_penalty

    # split_by_class = []
    # starting_latents_split_t = tf.convert_to_tensor(starting_latents_split)
    # for i in range(5): #i for class
    #     datum = tf.gather(starting_latents_split_t, input[i])
    #     datum_class = tf.gather(datum, i)  #max loss datum, class
    #     split_by_class.append(datum_class)
    # spurious = tf.concat([tf.expand_dims(split_by_class[0], 0), tf.expand_dims(split_by_class[1], 0)], 0)
    # spurious = tf.concat([spurious, tf.expand_dims(split_by_class[2], 0)], 0)
    # spurious = tf.concat([spurious, tf.expand_dims(split_by_class[3], 0)], 0)
    # spurious = tf.concat([spurious, tf.expand_dims(split_by_class[4], 0)], 0)

    spurious = starting_latents_split
    loss, theta = self.forward_decoder(data, latents, tf.zeros([5, 1, 640], dtype=tf.float32))
    for i in range(5): #temp simulate convergence #self._inner_unroll_length): #number of adaptation steps
      corr_penalty = tf.nn.l2_loss((latents-spurious))
      loss += 0.00001*corr_penalty
      loss_grad = tf.gradients(loss, latents)  # dLtrain/dz
      latents -= inner_lr * loss_grad[0]
      loss, classifier_weights = self.forward_decoder(data, latents, theta)

    adapted_latents, adapted_kl, adapted_kl_components, adapted_kl_zn = self.kl_divergence_given_latents(distribution_params, latents)

    # after adapting the latents, measure how large the divergence is (on average and for each component)
    # latents, adapted_kl, adapted_kl_components = self.kl_divergence_given_latents(distribution_params, latents)
    # mean_kl = tf.reduce_mean(adapted_kl)
    # mask = tf.cast(adapted_kl > mean_kl, tf.float32)
    # latents = tf.multiply(latents, mask)
    # latents = tf.clip_by_value(latents, clip_value_min=0., clip_value_max=1.)

    if self.is_meta_training:
    # stop_gradients lets you do the computation,
    # without updating it using sgd when the loss is taken wrt to the parameters
    #reduces the load of the adaptation procedure
      encoder_penalty = tf.losses.mean_squared_error(
          labels=tf.stop_gradient(latents), predictions=starting_latents)
      encoder_penalty = tf.cast(encoder_penalty, self._float_dtype)
    else:
      encoder_penalty = tf.constant(0., self._float_dtype)

    return loss, classifier_weights, encoder_penalty, corr_penalty, adapted_latents, adapted_kl, adapted_kl_components, adapted_kl_zn, spurious

  def get_split_features_full(self, data, method="none"):
    split_dim = int(64 / 8)
    split_data = []
    for i in range(8):
      start_idx = split_dim * i
      end_idx = split_dim * i + split_dim
      data_i = data[:, :, start_idx:end_idx]
      data_t = tf.nn.l2_normalize(data_i, axis=0)

      start_stack = 0
      end_stack = start_idx
      data_before = data[:, :, start_stack:end_stack]

      start_stack_after = end_idx
      end_stack_after = 64
      data_after = data[:, :, start_stack_after:end_stack_after]

      full = tf.concat([data_before, data_t, data_after], -1)
      split_data.append(full)
    return split_data

  def get_split_features(self, data, center, method="none"):
    split_dim = int(64 / 8)
    split_data = []
    for i in range(8):
      start_idx = split_dim * i
      end_idx = split_dim * i + split_dim
      data_i = data[:, :, start_idx:end_idx]
      if center is not None:
        center_i = center[:, :, start_idx:end_idx]
      else:
        center_i = None
      data_i = self.preprocess_split(data_i, center_i, method)
      split_data.append(data_i)
    return split_data

  def preprocess_split(self, data, center=None, method="none"):
    if method == "none":
      return data
    elif method == "l2n":
      return tf.nn.l2_normalize(data, axis=-1)
    elif method == "cl2n":
        data = tf.nn.l2_normalize(data, axis=-1)
        return tf.nn.l2_normalize(data - center, axis=-1)

  @snt.reuse_variables
  def finetuning_inner_loop(self, data, leo_loss, classifier_weights):
    tr_loss = leo_loss
    with tf.variable_scope("finetuning"):
      finetuning_lr = tf.get_variable(
          "lr", [1, 1, self.embedding_dim],
          dtype=self._float_dtype,
          initializer=tf.constant_initializer(self._finetuning_lr_init))
    #directly fine tune the weights to reduce the loss

    for _ in range(self._finetuning_unroll_length):
      loss_grad = tf.gradients(tr_loss, classifier_weights)
      classifier_weights -= finetuning_lr * loss_grad[0]
      tr_loss, _ = self.calculate_inner_loss(data.tr_input, data.tr_output,
                                             classifier_weights)

    val_loss, val_accuracy = self.calculate_inner_loss(
        data.val_input, data.val_output, classifier_weights)
    return val_loss, val_accuracy

  @snt.reuse_variables
  def forward_encoder(self, data):
    encoder_outputs = self.encoder(data.tr_input)
    relation_network_outputs = self.relation_network(encoder_outputs)
    latent_dist_params = self.average_codes_per_class(relation_network_outputs)
    latents, kl, kl_components, kl_zn = self.possibly_sample(latent_dist_params)
    return latents, kl, kl_components, kl_zn, latent_dist_params #temp

  @snt.reuse_variables
  def forward_decoder(self, data, latents, theta):
    weights_dist_params = self.decoder(latents)
    # Default to glorot_initialization and not stddev=1.
    fan_in = self.embedding_dim.value
    fan_out = self.num_classes.value
    stddev_offset = np.sqrt(2. / (fan_out + fan_in))
    classifier_weights, kl_for_weights, _, _ = self.possibly_sample(weights_dist_params,
                                                 stddev_offset=stddev_offset)
    # if (ifL2):
    tr_loss, _ = self.calculate_inner_loss_with_l2(data.tr_input, data.tr_output,
                                                   classifier_weights, theta)
    # tr_loss += kl_weights*0.01
    # else:
    #     tr_loss, _ = self.calculate_inner_loss(data.tr_input, data.tr_output,
    #                                                    classifier_weights)

    return tr_loss, classifier_weights

  @snt.reuse_variables
  def encoder(self, inputs):
    with tf.variable_scope("encoder"):
      after_dropout = tf.nn.dropout(inputs, rate=self.dropout_rate)
      regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
      initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
      encoder_module = snt.Linear(
          self._num_latents,
          use_bias=False,
          regularizers={"w": regularizer},
          initializers={"w": initializer},
      )
      outputs = snt.BatchApply(encoder_module)(after_dropout)
      return outputs

  @snt.reuse_variables
  def relation_network(self, inputs):
    with tf.variable_scope("relation_network"):
      regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
      initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
      relation_network_module = snt.nets.MLP(
          [2 * self._num_latents] * 3,
          use_bias=False,
          regularizers={"w": regularizer},
          initializers={"w": initializer},
      )
      total_num_examples = self.num_examples_per_class*self.num_classes
      inputs = tf.reshape(inputs, [total_num_examples, self._num_latents])

      left = tf.tile(tf.expand_dims(inputs, 1), [1, total_num_examples, 1])
      right = tf.tile(tf.expand_dims(inputs, 0), [total_num_examples, 1, 1])
      concat_codes = tf.concat([left, right], axis=-1)
      outputs = snt.BatchApply(relation_network_module)(concat_codes)
      outputs = tf.reduce_mean(outputs, axis=1)
      # 2 * latents, because we are returning means and variances of a Gaussian
      outputs = tf.reshape(outputs, [self.num_classes,
                                     self.num_examples_per_class,
                                     2 * self._num_latents])

      return outputs

  @snt.reuse_variables
  def decoder(self, inputs):
    with tf.variable_scope("decoder"):
      l2_regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
      orthogonality_reg = get_orthogonality_regularizer(
          self._orthogonality_penalty_weight)
      initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
      # 2 * embedding_dim, because we are returning means and variances
      decoder_module = snt.Linear(
          2 * self.embedding_dim,
          use_bias=False,
          regularizers={"w": l2_regularizer},
          initializers={"w": initializer},
      )
      outputs = snt.BatchApply(decoder_module)(inputs)
      self._orthogonality_reg = orthogonality_reg(decoder_module.w)
      return outputs

  def average_codes_per_class(self, codes):
    codes = tf.reduce_mean(codes, axis=1, keep_dims=True)  # K dimension
    # Keep the shape (N, K, *)
    codes = tf.tile(codes, [1, self.num_examples_per_class, 1])
    return codes

  def possibly_sample(self, distribution_params, stddev_offset=0.):
    means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1)
    stddev = tf.exp(unnormalized_stddev)
    stddev -= (1. - stddev_offset)
    stddev = tf.maximum(stddev, 1e-10)
    distribution = tfp.distributions.Normal(loc=means, scale=stddev)
    if not self.is_meta_training:
      return means, tf.constant(0., dtype=self._float_dtype),  tf.constant(0., dtype=self._float_dtype), tf.constant(0., dtype=self._float_dtype)
    # sampled latents for each class 5,1,64
    samples = distribution.sample()
    # 5, 1, 128 distribution_params
    # interpret each sample as a factor of a joint distribution over the latent variable z_n
    # 5,1,64 distributions (1 for each of the 64 means and std deviations)
    kl_divergence, kl_divergence_components, kl_divergence_zn = self.kl_divergence(samples, distribution)
    return samples, kl_divergence, kl_divergence_components, kl_divergence_zn

  def sample(self, distribution_params, stddev_offset=0.):
    means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1)
    stddev = tf.exp(unnormalized_stddev)
    stddev -= (1. - stddev_offset)
    stddev = tf.maximum(stddev, 1e-10)
    distribution = tfp.distributions.Normal(loc=means, scale=stddev)
    if not self.is_meta_training:
      return means
    # sampled latents for each class 5,1,64
    samples = []
    for i in range(8):
        samples.append(distribution.sample())
    samples = tf.concat([tf.expand_dims(t, 0) for t in samples], 0)
    return samples

  def kl_divergence_given_latents(self, distribution_params, adapted_latents, stddev_offset=0.):
    means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1)
    stddev = tf.exp(unnormalized_stddev)
    stddev -= (1. - stddev_offset)
    stddev = tf.maximum(stddev, 1e-10)
    distribution = tfp.distributions.Normal(loc=means, scale=stddev)
    if not self.is_meta_training:
      return means, tf.constant(0., dtype=self._float_dtype),  tf.constant(0., dtype=self._float_dtype), tf.constant(0., dtype=self._float_dtype)

    kl_divergence, kl_divergence_components, kl_divergence_zn = self.kl_divergence(adapted_latents, distribution)

    # prior_dist = tfd.MultivariateNormalDiag(loc=tf.zeros_like(adapted_latents),
    #                                         scale_diag=tf.ones_like(adapted_latents))
    # var_post_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=stddev)
    # kl_divergence = tfd.kl_divergence(distribution_a=var_post_dist, distribution_b=prior_dist)

    return adapted_latents, kl_divergence, kl_divergence_components, kl_divergence_zn

  # KL divergence of a multivariate gaussian posterior with a multivariate gaussian standard normal
  # identify which factors remain invariant
  # learn a distribution over which factors remain invariant
  # probablistic thresholding as intervening on the latent space
  def kl_divergence(self, samples, normal_distribution):
    random_prior = tfp.distributions.Normal(
        loc=tf.zeros_like(samples), scale=tf.ones_like(samples))
    #observation: preadaptation, the components are positive, postadaptation, they are negative. Why is that?
    # because the divergence is computed as the sum over sampled values. If it integrated over the entire support, it would be positive
    #returns the log of the probability density/mass function evaluated at the given sample value.
    kl_divergence_components = normal_distribution.log_prob(samples) - random_prior.log_prob(samples)
    # incorrect, this should be 5,1,1
    kl = tf.reduce_mean(kl_divergence_components)
    kl_divergence_zn = tf.reduce_mean(kl_divergence_components, axis=-1)
    return kl, kl_divergence_components, kl_divergence_zn

  def predict(self, inputs, weights):
    after_dropout = tf.nn.dropout(inputs, rate=self.dropout_rate)
    # This is 3-dimensional equivalent of a matrix product, where we sum over
    # the last (embedding_dim) dimension. We get [N, K, N, K] tensor as output.
    per_image_predictions = tf.einsum("ijk,lmk->ijlm", after_dropout, weights)

    # Predictions have shape [N, K, N]: for each image ([N, K] of them), what
    # is the probability of a given class (N)?
    predictions = tf.reduce_mean(per_image_predictions, axis=-1)
    return predictions

  #adjust the inner loss to include l2
  def calculate_inner_loss(self, inputs, true_outputs, classifier_weights):
    model_outputs = self.predict(inputs, classifier_weights)
    model_predictions = tf.argmax(
        model_outputs, -1, output_type=self._int_dtype)
    accuracy = tf.contrib.metrics.accuracy(model_predictions,
                                           tf.squeeze(true_outputs, axis=-1))

    return self.loss_fn(model_outputs, true_outputs), accuracy

  def calculate_inner_loss_with_l2(self, inputs, true_outputs, classifier_weights, theta):
      model_outputs = self.predict(inputs, classifier_weights)
      model_predictions = tf.argmax(
          model_outputs, -1, output_type=self._int_dtype)
      accuracy = tf.contrib.metrics.accuracy(model_predictions,
                                             tf.squeeze(true_outputs, axis=-1))

      return self.loss_fn_withl2(model_outputs, true_outputs, classifier_weights, theta), accuracy

  def save_problem_instance_stats(self, instance):
    num_classes, num_examples_per_class, embedding_dim = instance.get_shape()
    if hasattr(self, "num_classes"):
      assert self.num_classes == num_classes, (
          "Given different number of classes (N in N-way) in consecutive runs.")
    if hasattr(self, "num_examples_per_class"):
      assert self.num_examples_per_class == num_examples_per_class, (
          "Given different number of examples (K in K-shot) in consecutive"
          "runs.")
    if hasattr(self, "embedding_dim"):
      assert self.embedding_dim == embedding_dim, (
          "Given different embedding dimension in consecutive runs.")

    self.num_classes = num_classes
    self.num_examples_per_class = num_examples_per_class
    self.embedding_dim = embedding_dim

  @property
  def dropout_rate(self):
    return self._dropout_rate if self.is_meta_training else 0.0

  def loss_fn(self, model_outputs, original_classes):
    original_classes = tf.squeeze(original_classes, axis=-1)
    # Tensorflow doesn't handle second order gradients of a sparse_softmax yet.
    one_hot_outputs = tf.one_hot(original_classes, depth=self.num_classes)
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_outputs, logits=model_outputs)

  def loss_fn_withl2(self, model_outputs, original_classes, classifier_weights, theta):
    original_classes = tf.squeeze(original_classes, axis=-1)
    # Tensorflow doesn't handle second order gradients of a sparse_softmax yet.
    one_hot_outputs = tf.one_hot(original_classes, depth=self.num_classes)

    # if self.is_meta_training:
    # #remove correlations between generated weights
    #   decoder_penalty = tf.losses.mean_squared_error(
    #       labels=tf.stop_gradient(classifier_weights), predictions=theta)
    #   decoder_penalty = tf.cast(decoder_penalty, self._float_dtype)
    # else:
    #   decoder_penalty = tf.constant(0., self._float_dtype)

    return tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_outputs, logits=model_outputs)

    # observation: seems to be improving the accuracy, especially meta-train
    # return tf.nn.softmax_cross_entropy_with_logits_v2(
    #     labels=one_hot_outputs, logits=model_outputs) + self._encoder_penalty_weight*decoder_penalty

    # return tf.nn.softmax_cross_entropy_with_logits_v2(
    #     labels=one_hot_outputs, logits=model_outputs) + tf.nn.l2_loss(theta)


  def grads_and_vars(self, metatrain_loss):
    """Computes gradients of metatrain_loss, avoiding NaN.

    Uses a fixed penalty of 1e-4 to enforce only the l2 regularization (and not
    minimize the loss) when metatrain_loss or any of its gradients with respect
    to trainable_vars are NaN. In practice, this approach pulls the variables
    back into a feasible region of the space when the loss or its gradients are
    not defined.

    Args:
      metatrain_loss: A tensor with the LEO meta-training loss.

    Returns:
      A tuple with:
        metatrain_gradients: A list of gradient tensors.
        metatrain_variables: A list of variables for this LEO model.
    """
    metatrain_variables = self.trainable_variables
    metatrain_gradients = tf.gradients(metatrain_loss, metatrain_variables)

    nan_loss_or_grad = tf.logical_or(
        tf.is_nan(metatrain_loss),
        tf.reduce_any([tf.reduce_any(tf.is_nan(g))
                       for g in metatrain_gradients]))

    regularization_penalty = (
        1e-4 / self._l2_penalty_weight * self._l2_regularization)
    zero_or_regularization_gradients = [
        g if g is not None else tf.zeros_like(v)
        for v, g in zip(tf.gradients(regularization_penalty,
                                     metatrain_variables), metatrain_variables)]

    metatrain_gradients = tf.cond(nan_loss_or_grad,
                                  lambda: zero_or_regularization_gradients,
                                  lambda: metatrain_gradients, strict=True)

    return metatrain_gradients, metatrain_variables

  @property
  def _l2_regularization(self):
    return tf.cast(
        tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
        dtype=self._float_dtype)

  @property
  def _decoder_orthogonality_reg(self):
    return self._orthogonality_reg
