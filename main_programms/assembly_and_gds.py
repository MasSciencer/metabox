from metabox import modeling, assembly, export
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config

config.creat_folder(config.images_save_dir, True)
config.save_config(True)


# Load the metamodel we created in tutorial 1.
metamodel = modeling.load_metamodel(config.DNN_dir[0], config.DNN_dir[1])
# Define the bounds of the feature.
metamodel.set_feature_constraint("x_square",  vmin = config.const_rec_min, vmax = config.const_rec_X_max)
metamodel.set_feature_constraint('y_square', vmin = config.const_rec_Y_min, vmax = config.const_rec_max)

# Create a metasurface.
metasurface = assembly.Metasurface(
    diameter=config.diameter,
    refractive_index=config.refractive_index,
    thickness=config.thickness_MetaToSurface,
    metamodel=metamodel,          # the metamodel to use
    enable_propagator_cache=True, # cache the propagators for faster computation
    set_structures_variable=True, # set the structures as a variable to optimize
)

# Define the incidence wavelengths and angles.
incidence = assembly.Incidence(
    wavelength=np.linspace(config.wave_incident_start, config.wave_incident_end, config.split_broadwavelenght),
    phi=config.phi,
    theta=config.theta,
    jones_vector = config.jones_vector,
)

# Create a lens assembly.
lens_assembly = assembly.LensAssembly(
        surfaces=[metasurface], # Define the array of surfaces. Here only one.
        incidence=incidence,   # Define the incidence.
        figure_of_merit=config.figure_of_merit, # Define the figure of merit.
        use_x_pol=config.use_x_pol
)

# Use the Adam optimizer to optimize the lens assembly.
optimizer = tf.keras.optimizers.Adam(
    learning_rate=config.learning_rate
)
optimizer.build(lens_assembly.get_variables())

# Optimize the lens assembly. Returns the best-optimized lens assembly and the loss history.
history = assembly.optimize_single_lens_assembly(
    lens_assembly,
    optimizer,
    n_iter=config.n_iter,
    verbose=config.verbose,
    keep_best=config.keep_best,
)

# At this point, you can save the optimized metasurface to a file.
lens_assembly.save(config.assembly_dir[0], config.assembly_dir[1], overwrite=True)

# Plot the history of the figure of merit.
plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Figure of Merit")

# Compare the optimized and unoptimized fields.
#lens_assembly.show_color_psf(dir_save=config.images_save_dir) #where do you want to save?
lens_assembly.show_psf()

# You can access any surface in the assembly by calling the surfaces attribute.
# In this case, we only have one surface, so we can access it like so:
metasurface = lens_assembly.surfaces[0]
export.generate_gds(metasurface, 0, config.GDS_dir[0], config.GDS_dir[1])
print('____________GDS was generated____________')