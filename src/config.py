import os
import shutil
from metabox import rcwa, assembly
import numpy as np

# DON'T TOUCH ADJUSTMENT BELOW!!!!
# method for saving config into directory
def save_config(answer):
    if answer == True:
        shutil.copy(r'/home/eduser/pycharm/metabox/src/config.py', base_save_dir)
        print('config.py is copied to ' + base_save_dir)

    else:
        print('config.py is NOT copied')


def creat_folder(
        path: str = '/home/eduser/pycharm/metabox/storage/',
        overwrite: bool = False,
) -> None:
    if not os.path.exists(path):
        if overwrite is True:
            os.makedirs(path)
            print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")

# DONT TOUCH ADJUSTMENT ABOVE!!!!


# ***********************************************
# * _   _   _                                   *
# *| | (_) | |__    _ __    __ _   _ __   _   _ *
# *| | | | | '_ \  | '__|  / _` | | '__| | | | |*
# *| | | | | |_) | | |    | (_| | | |    | |_| |*
# *|_| |_| |_.__/  |_|     \__,_| |_|     \__, |*
# *                                       |___/ *
# ***********************************************

#Material
material_lattice = rcwa.Material('quartz')
material_substrate = rcwa.Material('quartz')

# Define RCWA simulation configuration
xy_harmonics = (5, 5)  # Fourier orders in x and y
resolution = 128  # grid resolution per periodicity
return_tensor = True  # return tensor instead of a SimulationResult object
minibatch_size = 10 # number of simulations to run in parallel

#There is a configuration of meta-atoms
#Quantity
amount = 40
periodicity = 5e-6 # 5 microns
#Shapes
shapes = []
#---------Rectangle------------
rectangle_min = 0
rectangle_max= periodicity * 1
const_rec_min = 0.5e-6 # 0.5 microns
const_rec_max = periodicity * 0.9
const_rec_X_max = 4e-6 # 3 microns
const_rec_Y_min = 4e-6 # 3 microns

#Rotation
deg_start = 0
deg_end = 0
rot_amount = 0
#---------Circle------------
circle_min = 0
const_circle_min = 50e-9
circle_max = periodicity/2
const_circle_max = periodicity/2 * 0.9

#Assembly
# x_width = rcwa.Feature(vmin=config.rectangle_min, vmax=config.rectangle_max, name="x_square", sampling = config.amount)
# y_width = rcwa.Feature(vmin=config.rectangle_min, vmax=config.rectangle_max, name="y_square", sampling = config.amount)
# Rectangle0 = rcwa.Rectangle(config.material_lattice, x_width = x_width, y_width = y_width, rotation_deg=0)
# Rectangle90 = rcwa.Rectangle(config.material_lattice, x_width = x_width, y_width = y_width, rotation_deg=90)
# config.shapes = [Rectangle0, Rectangle90]

#There is a configuration of Wave front
wave_start = 3.7e-6
wave_end = 4.8e-6
wave_step = 15

#There is configuration of Layers
thickness_cel = 5e-6 # 5 micron
thickness_surf = 1e-6 # 1 micron

# *************************
# * ____    _   _   _   _ *
# *|  _ \  | \ | | | \ | |*
# *| | | | |  \| | |  \| |*
# *| |_| | | |\  | | |\  |*
# *|____/  |_| \_| |_| \_|*
# *************************

# Input layer -> 10 (relu) -> 128 (tanh) -> 256 (relu) -> 256 (tanh) -> 128 (relu) -> 10 (tanh) -> Output layer
n_epochs = 150
hidden_layer_units_list = [10, 128, 256, 256, 256, 256, 256, 128, 10]
activation_list = ['relu', 'tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'tanh', 'tanh']
train_batch_size = 60


# ************************************************************
# *                                        _       _         *
# *  __ _   ___   ___    ___   _ __ ___   | |__   | |  _   _ *
# * / _` | / __| / __|  / _ \ | '_ ` _ \  | '_ \  | | | | | |*
# *| (_| | \__ \ \__ \ |  __/ | | | | | | | |_) | | | | |_| |*
# * \__,_| |___/ |___/  \___| |_| |_| |_| |_.__/  |_|  \__, |*
# *                                                    |___/ *
# ************************************************************

# Create a metasurface.
diameter=10e-3             # d = 10 mm in diameter
refractive_index=1.0        # the propagation medium after the metasurface
thickness_MetaToSurface=10e-3            # f = 1/2 the distance to the next surface)

# Define the incidence wavelengths and angles.
wave_incident_start = 3.7e-6
wave_incident_end = 4.8e-6
split_broadwavelenght = 3   # split wavelength range from lib on that amount of steps.
phi=[0]                     # normal incidence
theta=[0]                   # normal incidence
jones_vector = (1, 0)       # light with the electric field vector parallel to the x axis.

# Importain: evaluation of pol is not implemented
#   - Defaults to (1, 0) which corresponds to a linearly polarized
#   - Left CP (1/np.sqrt(2), 1j/np.sqrt(2))

# Create a lens assembly.
figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO # Define the figure of merit.
use_x_pol=True

# Use the Adam optimizer to optimize the lens assembly. This rate should be
# empirically determined.
learning_rate=1e-8

# Optimize the lens assembly. Returns the best-optimized lens assembly and the loss history.
n_iter=100
verbose=1
keep_best=True

# ···························································
# : ____    _____    ___    ____       _       ____   _____ :
# :/ ___|  |_   _|  / _ \  |  _ \     / \     / ___| | ____|:
# :\___ \    | |   | | | | | |_) |   / _ \   | |  _  |  _|  :
# : ___) |   | |   | |_| | |  _ <   / ___ \  | |_| | | |___ :
# :|____/    |_|    \___/  |_| \_\ /_/   \_\  \____| |_____|:
# ···························································

#There is a configuration for storage and saving
iteration = '2'
wavelength_tag = 'main_3.7_4.8'
shapes_tag = 'quartz_crosses'
base_dir = '/home/eduser/pycharm/metabox/storage'
base_save_dir = os.path.join(base_dir, wavelength_tag, iteration)

lib_dir = [shapes_tag + 'lib', base_save_dir]
DNN_dir = [shapes_tag + 'metamodel', base_save_dir]
assembly_dir = [shapes_tag + 'assembly', base_save_dir]
GDS_dir = [shapes_tag, base_save_dir]
images_save_dir = base_save_dir + '/color_psf'
