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
"""A binary building the graph and performing the optimization of LEO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pickle

from absl import flags
from six.moves import zip
import tensorflow as tf

import config
import data
import model
import utils

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import numpy as np

import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "/Users/zohra/PycharmProjects/leo/logs/latents-l2-all-spurL-r5", "Path to restore from and "
                    "save to checkpoints.")
flags.DEFINE_integer(
    "checkpoint_steps", 1000, "The frequency, in number of "
    "steps, of saving the checkpoints.")
flags.DEFINE_boolean("evaluation_mode", False, "Whether to run in an "
                     "evaluation-only mode.")
#to test set checkpoint_steps = None and evaluation_mode = True

def _clip_gradients(gradients, gradient_threshold, gradient_norm_threshold):
  """Clips gradients by value and then by norm."""
  if gradient_threshold > 0:
    gradients = [
        tf.clip_by_value(g, -gradient_threshold, gradient_threshold)
        for g in gradients
    ]
  if gradient_norm_threshold > 0:
    gradients = [
        tf.clip_by_norm(g, gradient_norm_threshold) for g in gradients
    ]
  return gradients


def _construct_validation_summaries(metavalid_loss, metavalid_accuracy, metavalid_generalization_loss):
  tf.summary.scalar("metavalid_loss", metavalid_loss)
  tf.summary.scalar("metavalid_valid_accuracy", metavalid_accuracy)
  # tf.summary.scalar("metavalid_generalization_loss", metavalid_generalization_loss)
  # The summaries are passed implicitly by TensorFlow.


def _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                  model_grads, model_vars, metatrain_generalization_loss,
                                  kl_components, adapted_kl_components, kl_zn, adapted_kl_zn, kl, adapted_kl, latents,
                                  adapted_latents, spurious):
  tf.summary.scalar("metatrain_loss", metatrain_loss)
  tf.summary.scalar("metatrain_valid_accuracy", metatrain_accuracy)
  # tf.summary.scalar("metavalid_generalization_loss", metatrain_generalization_loss)
  # tf.summary.scalar("metatrain_corr_penalty", metatrain_corr_penalty)

  for g, v in zip(model_grads, model_vars):
    histogram_name = v.name.split(":")[0]
    tf.summary.histogram(histogram_name, v)
    histogram_name = "gradient/{}".format(histogram_name)
    tf.summary.histogram(histogram_name, g)

  # class 5 + components 64
  # tf.summary.histogram("log(q(zn|x)/p(z)) - Initial", kl_components)
  # tf.summary.histogram("log(q(zn|x)/p(z)) - Adapted", adapted_kl_components)
  #
  # # class 5
  # tf.summary.histogram("KL-Divergence q(zn|x) || p(z) - Initial", kl_zn)
  # tf.summary.histogram("KL-Divergence q(zn|x) || p(z) - Adapted", adapted_kl_zn)
  #
  # batch
  # tf.summary.histogram("KL-Divergence - Initial", kl)
  # tf.summary.histogram("KL-Divergences - Adapted", adapted_kl)
  #
  # # class 5 + components 64
  tf.summary.histogram("Latents - Initial", latents)
  tf.summary.histogram("Latents - Adapted", adapted_latents)
  tf.summary.histogram("Latents - Spurious", spurious)

def _construct_examples_batch(batch_size, split, num_classes,
                              num_tr_examples_per_class,
                              num_val_examples_per_class):
  data_provider = data.DataProvider(split, config.get_data_config())
  examples_batch = data_provider.get_batch(batch_size, num_classes,
                                           num_tr_examples_per_class,
                                           num_val_examples_per_class)
  return utils.unpack_data(examples_batch)


def _construct_loss_and_accuracy(inner_model, inputs, is_meta_training):
  """Returns batched loss and accuracy of the model ran on the inputs."""
  call_fn = functools.partial(
      inner_model.__call__, is_meta_training=is_meta_training)
  per_instance_loss, per_instance_accuracy, per_instance_generalization_loss, \
  kl_components, adapted_kl_components, kl_zn, adapted_kl_zn, kl, adapted_kl, latents, adapted_latents, spurious = tf.map_fn(
      call_fn,
      inputs,
      dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
      back_prop=is_meta_training)
  loss = tf.reduce_mean(per_instance_loss)
  accuracy = tf.reduce_mean(per_instance_accuracy)
  return loss, accuracy, per_instance_generalization_loss, kl_components, adapted_kl_components, kl_zn, adapted_kl_zn, kl, adapted_kl, latents, adapted_latents, spurious


def construct_graph(outer_model_config):
  """Constructs the optimization graph."""
  inner_model_config = config.get_inner_model_config()
  tf.logging.info("inner_model_config: {}".format(inner_model_config))
  leo = model.LEO(inner_model_config, use_64bits_dtype=False)

  num_classes = outer_model_config["num_classes"]
  num_tr_examples_per_class = outer_model_config["num_tr_examples_per_class"]
  metatrain_batch = _construct_examples_batch(
      outer_model_config["metatrain_batch_size"], "train", num_classes,
      num_tr_examples_per_class,
      outer_model_config["num_val_examples_per_class"])
  metatrain_loss, metatrain_accuracy, metatrain_generalization_loss, \
  kl_components, adapted_kl_components, kl_zn, adapted_kl_zn, kl, adapted_kl, \
  latents, adapted_latents, spurious = _construct_loss_and_accuracy(
      leo, metatrain_batch, True) #returned by the inner_leo_loop

  metatrain_gradients, metatrain_variables = leo.grads_and_vars(metatrain_loss)

  # Avoids NaNs in summaries.
  metatrain_loss = tf.cond(tf.is_nan(metatrain_loss),
                           lambda: tf.zeros_like(metatrain_loss),
                           lambda: metatrain_loss)
  # adapted_kl_components = tf.cond(tf.is_nan(adapted_kl_components),
  #                          lambda: tf.zeros_like(adapted_kl_components),
  #                          lambda: adapted_kl_components)
  #
  # adapted_kl_zn = tf.cond(tf.is_nan(adapted_kl_zn),
  #                                 lambda: tf.zeros_like(adapted_kl_zn),
  #                                 lambda: adapted_kl_zn)
  # adapted_kl = tf.cond(tf.is_nan(adapted_kl),
  #                         lambda: tf.zeros_like(adapted_kl),
  #                         lambda: adapted_kl)
  # kl = tf.cond(tf.is_nan(kl),
  #                      lambda: tf.zeros_like(kl),
  #                      lambda: kl)
  metatrain_gradients = _clip_gradients(
      metatrain_gradients, outer_model_config["gradient_threshold"],
      outer_model_config["gradient_norm_threshold"])

  _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                metatrain_gradients, metatrain_variables, metatrain_generalization_loss,
                                kl_components, adapted_kl_components, kl_zn, adapted_kl_zn, kl, adapted_kl,  latents, adapted_latents, spurious)
  optimizer = tf.train.AdamOptimizer(
      learning_rate=outer_model_config["outer_lr"])
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.apply_gradients(
      list(zip(metatrain_gradients, metatrain_variables)), global_step)
  #after applying the gradients, compute the meta-validation loss using the same algorithm
  data_config = config.get_data_config()
  tf.logging.info("data_config: {}".format(data_config))
  total_examples_per_class = data_config["total_examples_per_class"]
  metavalid_batch = _construct_examples_batch(
      outer_model_config["metavalid_batch_size"], "val", num_classes,
      num_tr_examples_per_class,
      total_examples_per_class - num_tr_examples_per_class)
  metavalid_loss, metavalid_accuracy, metavalid_generalization_loss, _, _, _, _, _, _, _, _, _ = _construct_loss_and_accuracy(
      leo, metavalid_batch, False)

  metatest_batch = _construct_examples_batch(
      outer_model_config["metatest_batch_size"], "test", num_classes,
      num_tr_examples_per_class,
      total_examples_per_class - num_tr_examples_per_class)
  _, metatest_accuracy, _, _, _, _, _, _, _, _, _, _ = _construct_loss_and_accuracy(
      leo, metatest_batch, False)

  _construct_validation_summaries(metavalid_loss, metavalid_accuracy, metavalid_generalization_loss)

  return (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
          metatest_accuracy, kl_components, adapted_kl_components, kl_zn, adapted_kl_zn, kl, adapted_kl, latents, adapted_latents, spurious)


def run_training_loop(checkpoint_path):
  """Runs the training loop, either saving a checkpoint or evaluating it."""
  outer_model_config = config.get_outer_model_config()
  tf.logging.info("outer_model_config: {}".format(outer_model_config))
  (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
   metatest_accuracy, kl_components, adapted_kl_components, kl_zn, adapted_kl_zn, kl, adapted_kl, latents, adapted_latents, spurious) = construct_graph(outer_model_config)

  num_steps_limit = outer_model_config["num_steps_limit"]
  best_metavalid_accuracy = 0.

  # curate summary
  classes_seen = {}

  kl_components_hist = []
  adapted_kl_components_hist = []

  kl_zn_hist = []
  adapted_kl_zn_hist = []

  kl_hist = []
  adapted_kl_hist = []

  latents_hist = []
  metavalid_accuracy_hist = []

  for i in range(5):
      latents_hist.append([])

      kl_components_hist.append([])
      adapted_kl_components_hist.append([])

      kl_zn_hist.append([])
      adapted_kl_zn_hist.append([])

      for j in range(64):
        kl_components_hist[i].append([])
        adapted_kl_components_hist[i].append([])
        latents_hist[i].append([])


  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_path,
      save_summaries_steps=FLAGS.checkpoint_steps,
      log_step_count_steps=FLAGS.checkpoint_steps,
      save_checkpoint_steps=FLAGS.checkpoint_steps,
      summary_dir=checkpoint_path) as sess:
      # hooks=[wandb.tensorflow.WandbHook(steps_per_log=10)]) as sess:
    if not FLAGS.evaluation_mode:
      global_step_ev = sess.run(global_step)
      while global_step_ev < num_steps_limit:
        if global_step_ev % FLAGS.checkpoint_steps == 0:
          # Just after saving checkpoint, calculate accuracy 10 times and save
          # the best checkpoint for early stopping.
          metavalid_accuracy_ev = utils.evaluate_and_average(
              sess, metavalid_accuracy, 1)  #runs the session for validation

          # kl_components_ev = utils.evaluate(sess, kl_components)
          # adapted_kl_components_ev = utils.evaluate(sess, adapted_kl_components)
          #
          # kl_zn_ev = utils.evaluate(sess, kl_zn)
          # adapted_kl_zn_ev = utils.evaluate(sess, adapted_kl_zn)
          #
          # # why is there only one kl divergence score for eatch batch. The divergence should be per class per component.
          #
          # kl_ev = utils.evaluate(sess, kl)
          # adapted_kl_ev = utils.evaluate(sess, adapted_kl)
          #
          latents_ev = utils.evaluate(sess, latents)
          adapted_latents_ev = utils.evaluate(sess, adapted_latents)
          spurious_ev = utils.evaluate(sess, spurious)

          # for batch in kl_components_ev:
          #     for c in batch:
          #         for components in c:
          #             for i, component in enumerate(components):
          #                 cl = int(component[0])
          #                 kl_val = component[1]
          #                 if (cl <= 5):  # collect data for sampled classes
          #                     # for each component
          #                     kl_components_hist[cl][i].append(kl_val)
          #                 if cl not in classes_seen:
          #                     classes_seen[cl] = 1
          #
          # for batch in adapted_kl_components_ev:
          #     for c in batch:
          #         for components in c:
          #             for i, component in enumerate(components):
          #                 cl = int(component[0])
          #                 kl_val = component[1]
          #                 if (cl <= 5):  # collect data for sampled classes
          #                     # for each class and component
          #                     adapted_kl_components_hist[cl][i].append(kl_val)
          #
          # for batch in kl_zn_ev: # batch, 5, 2
          #     for component in batch:
          #         cl = int(component[0])
          #         kl_zn_val = component[1]
          #         if (cl <= 5):  # collect data for sampled classes
          #             kl_zn_hist[cl].append(kl_zn_val)
          #
          # for batch in adapted_kl_zn_ev:  # batch, 5, 2
          #     for component in batch:
          #         cl = int(component[0])
          #         adapted_kl_zn_val = component[1]
          #         if (cl <= 5):  # collect data for sampled classes
          #             adapted_kl_zn_hist[cl].append(adapted_kl_zn_val)
          #
          for batch_change, batch_latents in zip(latents_ev - spurious_ev, latents_ev):
              for k, c in enumerate(batch_change):
                  for j, components in enumerate(c):
                      for i, component in enumerate(components):
                          cl = int(batch_latents[k][j][i][0])
                          latent_val = component[1]
                          if (cl <= 5):  # collect data for sampled classes
                              latents_hist[cl][i].append(latent_val)
          #
          # ########## Visualize kl history
          # _, ax = plt.subplots(5, 2, sharex='col', sharey='row', figsize=(20, 20))
          #
          # for i in range(5):
          #     color = iter(cm.rainbow(np.linspace(0, 1, 64)))
          #     for j in range(64):
          #         c = next(color)
          #         val = kl_components_hist[i][j]
          #         step = range(global_step_ev, global_step_ev+len(val))
          #         ax[i][0].plot(step, val, c=c) #adds values for each component using a different color
          #         ax[i][1].plot(step, adapted_kl_components_hist[i][j], c=c)
          #
          #     ax[i][0].set_title('N=' + str(i) + ' log(q(zn|x) / p(z)) ratio for Initial Factors')
          #     ax[i][1].set_title('N=' + str(i) + ' log(q(zn|x) / p(z)) ratio for Adapted Factors')
          #     # ax[i][0].legend(list(range(64)))
          #     ax[i][0].set_ylabel('kl divergence')
          #
          # ax[4][0].set_xlabel('step')
          # ax[4][1].set_xlabel('step')
          #
          # ######### Visualize kl_zn history
          # _, ax_zn = plt.subplots(5, 2, sharex='col', sharey='row', figsize=(20, 20))
          #
          # for i in range(5):
          #     color = iter(cm.rainbow(np.linspace(0, 1, 5)))
          #     c = next(color)
          #     val = kl_zn_hist[i]
          #     step = range(global_step_ev, global_step_ev+ len(val))
          #     ax_zn[i][0].plot(step, val, c=c)
          #     ax_zn[i][1].plot(step, adapted_kl_zn_hist[i], c=c)
          #
          #     ax_zn[i][0].set_title('N=' + str(i) + ' KL Divergence for Initial Zn for q(zn|x) and p(z)')
          #     ax_zn[i][1].set_title('N=' + str(i) + ' KL Divergence for Adapted Zn for q(zn|x) and p(z)')
          #
          # ax_zn[4][0].set_xlabel('step')
          # ax_zn[4][1].set_xlabel('step')
          #
          # ########### Visualize kl divergence for batches
          # kl_hist.append(kl_ev.flatten())
          # adapted_kl_hist.append(adapted_kl_ev.flatten())
          # _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
          # ax1.plot(range(global_step_ev, global_step_ev+ len(kl_hist)), kl_hist)
          # ax1.set_title('KL Divergence for Initial q(z|x) and p(z)')
          # ########### Visualize adapted kl divergence for batches
          # ax2.plot(range(global_step_ev, global_step_ev+ len(adapted_kl_hist)), adapted_kl_hist)
          # ax2.set_title('KL Divergence for Adapted q(z|x) and p(z)')
          #
          # metavalid_accuracy_hist.append(metavalid_accuracy_ev)
          # _, metavalid_accuracy_plot = plt.subplots()
          # metavalid_accuracy_plot.plot(range(0, len(metavalid_accuracy_hist)), metavalid_accuracy_hist)
          # metavalid_accuracy_plot.set_title('Metavalidation Accuracy')
          #
          #
          # Visualize latent history, additionally examine the gradients for the latents
          _, ax_latent = plt.subplots(5, sharex='col', figsize=(20, 20))

          for i in range(5):
              color = iter(cm.rainbow(np.linspace(0, 1, 64)))
              for j in range(64):
                  c = next(color)
                  step = range(0, len(latents_hist[i][j]))
                  ax_latent[i].plot(step, latents_hist[i][j], c=c)

              ax_latent[i].set_title('class=' + str(i) + ' Change in Latents')

          ax_latent[4].set_xlabel('step')
          plt.show();


          tf.logging.info("Step: {} meta-valid accuracy: {}".format(
              global_step_ev, metavalid_accuracy_ev))

          if metavalid_accuracy_ev > best_metavalid_accuracy:
            utils.copy_checkpoint(checkpoint_path, global_step_ev,
                                  metavalid_accuracy_ev)
            best_metavalid_accuracy = metavalid_accuracy_ev

        _, global_step_ev, metatrain_accuracy_ev = sess.run(
            [train_op, global_step, metatrain_accuracy]) #runs the session for training

        if global_step_ev % (FLAGS.checkpoint_steps // 2) == 0:
          tf.logging.info("Step: {} meta-train accuracy: {}".format(
              global_step_ev, metatrain_accuracy_ev))
    else:
      assert not FLAGS.checkpoint_steps
      num_metatest_estimates = (
          10000 // outer_model_config["metatest_batch_size"])

      test_accuracy = utils.evaluate_and_average(sess, metatest_accuracy,
                                                 num_metatest_estimates) #runs the session for testing

      tf.logging.info("Metatest accuracy: %f", test_accuracy)
      with tf.gfile.Open(
          os.path.join(checkpoint_path, "test_accuracy"), "wb") as f:
        pickle.dump(test_accuracy, f)

def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
  """Plot a forecast distribution against the 'true' time series."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1, 1, 1)

  num_steps = len(y)
  num_steps_forecast = forecast_mean.shape[-1]
  num_steps_train = num_steps - num_steps_forecast


  ax.plot(x, y, lw=2, color=c1, label='ground truth')

  forecast_steps = np.arange(
      x[num_steps_train],
      x[num_steps_train]+num_steps_forecast,
      dtype=x.dtype)

  ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

  ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
           label='forecast')
  ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color=c2, alpha=0.2)

  ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
  yrange = ymax-ymin
  ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
  ax.set_title("{}".format(title))
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()

  return fig, ax

def plot_one_step_predictive(dates, observed_time_series,
                             one_step_mean, one_step_scale,
                             x_locator=None, x_formatter=None):
  """Plot a time series against a model's one-step predictions."""

  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  fig=plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1,1,1)
  num_timesteps = one_step_mean.shape[-1]
  ax.plot(dates, observed_time_series, label="observed time series", color=c1)
  ax.plot(dates, one_step_mean, label="one-step prediction", color=c2)
  ax.fill_between(dates,
                  one_step_mean - one_step_scale,
                  one_step_mean + one_step_scale,
                  alpha=0.1, color=c2)
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()
  fig.tight_layout()
  return fig, ax

def main(argv):
  del argv  # Unused.
  #wandb.init(project="visualize-leo", entity="zohra", config=tf.flags.FLAGS, sync_tensorboard=True)
  run_training_loop(FLAGS.checkpoint_path)


if __name__ == "__main__":
  tf.app.run()

