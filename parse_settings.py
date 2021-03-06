import json

data = {
    'master': '',
    'train_dir': '/tmp/tfmodel/',
    'num_clones': 1,
    'clone_on_cpu': False,
    'worker_replicas': 1,
    'num_ps_tasks': 0,
    'num_readers': 4,
    'num_preprocessing_threads': 4,
    'log_every_n_steps': 10,
    'save_summaries_secs': 600,
    'save_interval_secs': 600,
    'task': 0,
    'weight_decay': 0.00004,
    'optimizer': 'rmsprop',
    'adadelta_rho': 0.95,
    'adagrad_initial_accumulator_value': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'opt_epsilon': 1.0,
    'ftrl_learning_rate_power': -0.5,
    'ftrl_initial_accumulator_value': 0.1,
    'ftrl_l1': 0.0,
    'ftrl_l2': 0.0,
    'momentum': 0.9,
    'rmsprop_momentum': 0.9,
    'rmsprop_decay': 0.9,
    'learning_rate_decay_type': 'exponential',
    'learning_rate': 0.01,
    'end_learning_rate': 0.0001,
    'label_smoothing': 0.0,
    'learning_rate_decay_factor': 0.94,
    'num_epochs_per_decay': 2.0,
    'sync_replicas': False,
    'replicas_to_aggregate': 1,
    'moving_average_decay': None,
    'dataset_name': 'cifar10',
    'dataset_split_name': 'train',
    'dataset_dir': '/tmp/cifar10/',
    'labels_offset': 0,
    'model_name': 'resnet_v1_50',
    'preprocessing_name': None,
    'batch_size': 32,
    'train_image_size': None,
    'max_number_of_steps': None,
    'max_num_batches': None,
    'checkpoint_path': None,
    'checkpoint_exclude_scopes': None,
    'trainable_scopes': None,
    'ignore_missing_vars': False,
    'attention_module': None,
    'eval_dir': '/tmp/tfmodel/',
    'eval_image_size': None,
    'eval_interval_secs': 600
}

with open('settings.json', 'w') as settings:
    json.dump(data, settings, indent=2)