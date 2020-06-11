# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

with open('settings.json') as f:
    data = json.load(f)

def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        data['dataset_name'], data['dataset_split_name'], data['dataset_dir'])

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        data['model_name'],
        num_classes=(dataset.num_classes - data['labels_offset']),
        is_training=False,
        attention_module=data['attention_module'])

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * data['batch_size'],
        common_queue_min=data['batch_size'])
    [image, label] = provider.get(['image', 'label'])
    label -= data['labels_offset']

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = data['preprocessing_name'] or data['model_name']
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = data['eval_image_size'] or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=data['batch_size'],
        num_threads=data['num_preprocessing_threads'],
        capacity=5 * data['batch_size'])

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if data['moving_average_decay']:
      variable_averages = tf.train.ExponentialMovingAverage(
          data['moving_average_decay, tf_global_step'])
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    summary_ops = []
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
      summary_ops.append(op)

    # TODO(sguada) use num_epochs=1
    if data['max_num_batches']:
      num_batches = data['max_num_batches']
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(data['batch_size']))
    """
    if tf.gfile.IsDirectory(data['checkpoint_path']):
      checkpoint_path = tf.train.latest_checkpoint(data['checkpoint_path'])
    else:
      checkpoint_path = data['checkpoint_path']
    """

    tf.logging.info('Evaluating %s' % data['checkpoint_path'])

    # GPU memory dynamic allocation
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    slim.evaluation.evaluation_loop(
    master=data['master'],
    checkpoint_dir=data['checkpoint_path'],
    logdir=data['eval_dir'],
    num_evals=num_batches,
    eval_op=list(names_to_updates.values()),
    summary_op=tf.summary.merge(summary_ops),
    eval_interval_secs=data['eval_interval_secs'],
    variables_to_restore=variables_to_restore,
    session_config=session_config)
    """
    slim.evaluation.evaluate_once(
        master=data['master'],
        checkpoint_path=checkpoint_path,
        logdir=data['eval_dir'],
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)
    """    


if __name__ == '__main__':
  tf.app.run()
