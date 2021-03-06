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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import os

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.tools import freeze_graph

slim = tf.contrib.slim

with open('settings.json') as f:
    data = json.load(f);


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    Raises:
      ValueError: if
    """
    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch data['num_epochs_per_decay] times. This is different
    # behavior from sync replicas and is expected to produce different results.
    decay_steps = int(num_samples_per_epoch * data['num_epochs_per_decay'] /
                      data['batch_size'])

    if data['sync_replicas']:
        decay_steps /= data['replicas_to_aggregate']

    if data['learning_rate_decay_type'] == 'exponential':
        return tf.train.exponential_decay(data['learning_rate'],
                                          global_step,
                                          decay_steps,
                                          data['learning_rate_decay_factor'],
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif data['learning_rate_decay_type'] == 'fixed':
        return tf.constant(data['learning_rate'], name='fixed_learning_rate')
    elif data['learning_rate_decay_type'] == 'polynomial':
        return tf.train.polynomial_decay(data['learning_rate'],
                                         global_step,
                                         decay_steps,
                                         data['end_learning_rate'],
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         data['learning_rate_decay_type'])




def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    Raises:
      ValueError: if data['optimizer'] is not recognized.
    """
    if data['optimizer'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=data['adadelta_rho'],
            epsilon=data['opt_epsilon'])
    elif data['optimizer'] == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=data['adagrad_initial_accumulator_value'])
    elif data['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=data['adam_beta1'],
            beta2=data['adam_beta2'],
            epsilon=data['opt_epsilon'])
    elif data['optimizer'] == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=data['ftrl_learning_rate_power'],
            initial_accumulator_value=data['ftrl_initial_accumulator_value'],
            l1_regularization_strength=data['ftrl_l1'],
            l2_regularization_strength=data['ftrl_l2'])
    elif data['optimizer'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=data['momentum'],
            name='Momentum')
    elif data['optimizer'] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=data['rmsprop_decay'],
            momentum=data['rmsprop_momentum'],
            epsilon=data['opt_epsilon'])
    elif data['optimizer'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % data['optimizer'])
    return optimizer


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.
    Returns:
      An init function run by the supervisor.
    """
    if data['checkpoint_path'] is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(data['train_dir']):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % data['train_dir'])
        return None

    exclusions = []
    if data['checkpoint_exclude_scopes']:
        exclusions = [scope.strip()
                      for scope in data['checkpoint_exclude_scopes'].split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(data['checkpoint_path']):
        checkpoint_path = tf.train.latest_checkpoint(data['checkpoint_path'])
    else:
        checkpoint_path = data['checkpoint_path']

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=data['ignore_missing_vars'])


def _get_variables_to_train():
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if data['trainable_scopes'] is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in data['trainable_scopes'].split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(_):
    if not data['dataset_dir']:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=data['num_clones'],
            clone_on_cpu=data['clone_on_cpu'],
            replica_id=data['task'],
            num_replicas=data['worker_replicas'],
            num_ps_tasks=data['num_ps_tasks'])

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            data['dataset_name'], data['dataset_split_name'], data['dataset_dir'])

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            data['model_name'],
            num_classes=(dataset.num_classes - data['labels_offset']),
            weight_decay=data['weight_decay'],
            is_training=True,
            attention_module=data['attention_module'])

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = data['preprocessing_name'] or data['model_name']
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=data['num_readers'],
                common_queue_capacity=20 * data['batch_size'],
                common_queue_min=10 * data['batch_size'])
            [image, label] = provider.get(['image', 'label'])
            label -= data['labels_offset']

            train_image_size = data['train_image_size'] or network_fn.default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=data['batch_size'],
                num_threads=data['num_preprocessing_threads'],
                capacity=5 * data['batch_size'])
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes - data['labels_offset'])
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels = batch_queue.dequeue()
            logits, end_points = network_fn(images)

            #############################
            # Specify the loss function #
            #############################
            if 'AuxLogits' in end_points:
                slim.losses.softmax_cross_entropy(
                    end_points['AuxLogits'], labels,
                    label_smoothing=data['label_smoothing'], weights=0.4,
                    scope='aux_loss')
            slim.losses.softmax_cross_entropy(
                logits, labels, label_smoothing=data['label_smoothing'], weights=1.0)
            return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if data['moving_average_decay']:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                data['moving_average_decay'], global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
            optimizer = _configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if data['sync_replicas']:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=data['replicas_to_aggregate'],
                total_num_replicas=data['worker_replicas'],
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables)
        elif data['moving_average_decay']:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train()

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')


        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=data['train_dir'],
            master=data['master'],
            is_chief=(data['task'] == 0),
            init_fn=_get_init_fn(),
            summary_op=summary_op,
            number_of_steps=data['max_number_of_steps'],
            log_every_n_steps=data['log_every_n_steps'],
            save_summaries_secs=data['save_summaries_secs'],
            save_interval_secs=data['save_interval_secs'],
            sync_optimizer=optimizer if data['sync_replicas'] else None)

if __name__ == '__main__':
    tf.app.run()