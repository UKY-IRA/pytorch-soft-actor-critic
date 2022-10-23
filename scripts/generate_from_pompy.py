from pompy import models, processors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import random
import argparse

# Seed random number generator
# seed = 20180517
# rng = np.random.RandomState(seed)

''' Parameters
        sim_region : Rectangle
            Two-dimensional rectangular region over which to model wind
            velocity field.
        n_x : integer
            Number of grid points in x direction.
        n_y : integer
            Number of grid points in y direction.
        u_av : float
            Mean x-component of wind velocity (dimension: length / time).
        v_av : float
            Mean y-component of wind velocity (dimension: length / time).
        k_x : float or array_like
            Diffusivity constant in x direction. Either a single scalar value
            across the whole simulated region or an array of size `(n_x, n_y)`
            defining values for each grid point (dimension: length**2 / time).
        k_y : float or array_like
            Diffusivity constant in y direction. Either a single scalar value
            across the whole simulated region or an array of size `(n_x, n_y)`
            defining values for each grid point (dimension: length**2 / time).
        noise_gain : float
            Input gain constant for boundary condition noise generation
            (dimensionless).
        noise_damp : float
            Damping ratio for boundary condition noise generation
            (dimensionless).
        noise_bandwidth : float
            Bandwidth for boundary condition noise generation (dimension:
            angle / time).
        use_original_noise_updates : boolean
            Whether to use the original non-SDE based updates for the noise
            process as defined in Farrell et al. (2002), see notes in
            `ColouredNoiseGenerator` documentation.
        sim_region : Rectangle
            2D rectangular region of space over which the simulation is
            conducted. This should be a subset of the simulation region defined
            for the wind model.
        source_pos : float iterable
            Coordinates of the fixed source position within the simulation
            region from which puffs are released. If a length 2 iterable is
            passed, the z coordinate will be set a default of 0
            (dimension: length).
        wind_model : WindModel
            Dynamic model of the large scale wind velocity field in the
            simulation region.
        model_z_disp : boolean
            Whether to model dispersion of puffs from plume centre-line in z
            direction. If set `True` then the puffs will be modelled as
            dispersing in the vertical direction by a random walk process (the
            wind model is limited to 2D hence the vertical wind speed is
            assumed to be zero), if set `False` the puff z-coordinates will not
            be updated from their initial value of 0.
        centre_rel_diff_scale : float or float iterable
            Scaling for the stochastic process used to model the centre-line
            relative diffusive transport of puffs. Either a single float value
            of isotropic diffusion in all directions, or one of a pair of
            values specifying different scales for the x and y directions
            respectively if `model_z_disp=False` or a triplet of values
            specifying different scales for x, y and z scales respectively if
            `model_z_disp=True` (dimension: length / time**0.5).
        puff_init_rad: float
            Initial radius of the puffs (dimension: length).
        puff_spread_rate : float
            Constant which determines the rate at which the odour puffs
            increase in size over time (dimension: length**2 / time).
        puff_release_rate : float
            Mean rate at which new puffs are released into the plume. Puff
            release is modelled as a stochastic Poisson process, with each puff
            released assumed to be independent and the mean release rate fixed
            (dimension: count/time).
        init_num_puffs : integer
            Initial number of puffs to release at the beginning of the
            simulation.
        max_num_puffs : integer
            Maximum number of puffs to permit to be in existence simultaneously
            within model, used to limit memory and processing requirements of
            model. This parameter needs to be set carefully in relation to the
            puff release rate and simulation region size as if too small it
            will lead to breaks in puff release when the number of puffs
            remaining in the simulation region reaches the limit.
'''
# TODO: standardized xdim, ydim, etc. from a params file

# Define wind model simulation region
wind_region = models.Rectangle(x_min=0., x_max=100., y_min=-50., y_max=50.)

# Define wind model parameters
xdim = 50 
ydim = 100
wind_model_params = { 
    'n_x': xdim,
    'n_y': ydim,
    'u_av': 0.,
    'v_av': 0.,
    'k_x': 40.,
    'k_y': 40.,
    'noise_gain': 20.,
    'noise_damp': 1.0,
    'noise_bandwidth': 0.3,
    'use_original_noise_updates': True
}

# Create wind model object
wind_model = models.WindModel(wind_region, **wind_model_params)

# Define plume simulation region
# This is a subset of the wind simulation region
sim_region = models.Rectangle(x_min=0., x_max=float(xdim), y_min=0., y_max=float(ydim))

# Define plume model Parameters
source_pos = (random.random()*(xdim/2)+xdim/4, random.random()*(ydim/2)+ydim/4, 0)
plume_model_params = {
    'source_pos': source_pos,
    'centre_rel_diff_scale': 2.,
    'puff_release_rate': 25,
    'puff_init_rad': 0.01**0.5,
    'puff_spread_rate': 0.025,
    'init_num_puffs': 100,
    'max_num_puffs': 2500,
    'model_z_disp': True,
}

# Create plume model object
plume_model = models.PlumeModel(
    sim_region=sim_region, wind_model=wind_model, **plume_model_params)

# Define concentration array (image) generator parameters
array_gen_params = {
    'array_z': 0.,
    'n_x': 50,
    'n_y': 100,
    'puff_mol_amount': 8.3e8
}

# Create concentration array generator object
array_gen = processors.ConcentrationArrayGenerator(
    array_xy_region=sim_region, **array_gen_params)
# Set up figure
fig = plt.figure(figsize=(2.5, 5))
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis('off')

# Display initial concentration field as image
conc_array = array_gen.generate_single_array(plume_model.puff_array)
conc_im = ax.imshow(
    conc_array.T, extent=sim_region, vmin=0., vmax=1e10, cmap='Reds')

# Simulation timestep
dt = 0.02

# Run wind model forward to equilibrate
for k in range(2000):
    wind_model.update(dt)

# Define animation update function
def update(i):
    # Do 10 time steps per frame update
    for k in range(10):
        wind_model.update(dt)
        plume_model.update(dt)
    conc_array = array_gen.generate_single_array(plume_model.puff_array)
    return conc_array
    # conc_im.set_data(conc_array.T)
    # return [conc_im]


frames = 500
wind_speeds = np.zeros((frames, xdim, ydim, 2))
gas_mask = np.zeros((frames, xdim, ydim))
for i in range(frames):
    gas_mask[i] = update(i)
    wind_speeds[i] = wind_model.velocity_field
map_over_time = np.concatenate((wind_speeds, gas_mask.reshape(frames, xdim, ydim, 1)), axis=3)
# WARNING: these are very big files, can blow up quick
parser = argparse.ArgumentParser(description='Generates a numpy array of flowfield frames generated by pompy.')
parser.add_argument('--prefix', '-p', default="example", help='Prefix to npy file output.')
args = parser.parse_args()
np.save(f"{args.prefix}_plume.npy", map_over_time, allow_pickle=False)

'''
# Animate plume concentration and save as MP4
anim = FuncAnimation(fig, update, frames=frames, repeat=False)
writergif = animation.PillowWriter(fps=30)
anim.save('plume.gif', writer=writergif)
'''
