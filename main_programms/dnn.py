from metabox import modeling
import tensorflow as tf
import config

# GPU_test
if tf.config.experimental.list_physical_devices('GPU'):
    try:
        # Настройте TensorFlow для использования всех доступных GPU
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(logical_gpus), "Logical GPUs are available for use.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. TensorFlow will use CPU for computations.")

# Now that we have saved our simulations, we can load them back in
# and use them to train a metamodel neural network
loaded_sim_lib = modeling.load_simulation_library(config.lib_dir[0], config.lib_dir[1])

# Let's train a DNN with the following architecture:
model = modeling.create_and_train_model(
    loaded_sim_lib,
    n_epochs=config.n_epochs,
    hidden_layer_units_list = config.hidden_layer_units_list,
    activation_list = config.activation_list,
    train_batch_size = config.train_batch_size,
)

model.save(config.DNN_dir[0], config.DNN_dir[1], overwrite=True)
model.plot_training_history()




