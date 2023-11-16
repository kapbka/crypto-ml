import logging

import tensorflow as tf


def limit_memory(max_bytes: int):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return

    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=max_bytes)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    logging.info(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
