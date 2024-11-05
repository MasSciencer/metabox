import tensorflow as tf

# GPU_test
if tf.config.experimental.list_physical_devices('GPU'):
    try:
        # Настройте TensorFlow для использования всех доступных GPU
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(logical_gpus), "Logical GPUs are available for use. Hurray!!!")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. TensorFlow will use CPU for computations.")