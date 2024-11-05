from metabox import rcwa, utils, modeling, export
import numpy as np
import config

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

Rectangle0 = rcwa.Rectangle(config.material_lattice, x_width = 5e-9, y_width = 100e-9, rotation_deg=45)
Rectangle90 = rcwa.Rectangle(config.material_lattice, x_width = 5e-9, y_width = 100e-9, rotation_deg=-45)
shapes = [Rectangle0, Rectangle90]

#Summarization of patterns
patterned_layer = rcwa.Layer(material=1, thickness=config.thickness_cel, shapes=shapes)
substrate = rcwa.Layer(material=config.material_substrate, thickness=config.thickness_surf)
cell = rcwa.UnitCell(
    layers=[patterned_layer, substrate],
    periodicity=periodicity,
)

cell_shape = export.unit_cell_to_gds_shape(cell, layer = 0)
export.unit_cell_to_svg(cell_shape, path = '/home/eduser/pycharm/metabox/storage/test/main.svg')
export.save_shape_gds(cell_shape, '/home/eduser/pycharm/metabox/storage/test', 'cell_test')