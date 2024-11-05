from metabox import assembly, modeling
import numpy as np
import config

config.creat_folder(config.images_save_dir, True)
config.save_config(False)

# For a new session, you can load the optimized metasurface from a file again.
loaded_assembly = assembly.load_lens_assembly(config.assembly_dir[0], config.assembly_dir[1])
metasurface = loaded_assembly.surfaces[0]


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

# Compare the optimized and unoptimized fields.
lens_assembly.show_color_psf(dir_save=config.images_save_dir) #where do you want to save?
lens_assembly.show_psf()
