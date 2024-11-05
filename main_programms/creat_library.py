from metabox import rcwa, utils, modeling
import numpy as np
import config

config.creat_folder(config.base_save_dir, overwrite=True)
config.save_config(True)

# Define sampling wavelengths
# In practice, you may want to simulate more wavelengths for better sampling density.
incidence = utils.Incidence(wavelength=np.linspace(config.wave_start, config.wave_end, config.wave_step))

# Define the unit cell periodicity
periodicity = (config.periodicity, config.periodicity)

# Define RCWA simulation configuration
sim_config = rcwa.SimConfig(
    xy_harmonics=config.xy_harmonics, # Fourier orders in x and y
    resolution=config.resolution, # grid resolution per periodicity
    return_tensor=config.return_tensor, # return tensor instead of a SimulationResult object
    minibatch_size=config.minibatch_size # number of simulations to run in parallel
)

#Let's creat a complex pattern of meta-atom
#Featuares are here and #Assembly square
#Assembly
x_width = rcwa.Feature(vmin=config.rectangle_min, vmax=config.rectangle_max, name="x_square", sampling = config.amount)
y_width = rcwa.Feature(vmin=config.rectangle_min, vmax=config.rectangle_max, name="y_square", sampling = config.amount)
Rectangle0 = rcwa.Rectangle(config.material_lattice, x_width = x_width, y_width = y_width, rotation_deg=0)
Rectangle90 = rcwa.Rectangle(config.material_lattice, x_width = x_width, y_width = y_width, rotation_deg=90)
config.shapes = [Rectangle0, Rectangle90]

#Summarization of patterns
patterned_layer = rcwa.Layer(material=1, thickness=config.thickness_cel, shapes=config.shapes)
substrate = rcwa.Layer(material=config.material_substrate, thickness=config.thickness_surf)
cell = rcwa.UnitCell(
    layers=[patterned_layer, substrate],
    periodicity=periodicity,
)

protocell = rcwa.ProtoUnitCell(cell)
sim_lib = modeling.sample_protocell(
    protocell=protocell,
    incidence=incidence,
    sim_config=sim_config,
)

sim_lib.save(config.lib_dir[0], config.lib_dir[1], overwrite=True)